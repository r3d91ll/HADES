package proxies

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"golang.org/x/net/http2"
)

const (
	defaultUpstreamSocket = "/run/arangodb3/arangodb.sock"
	defaultROListenSocket = "/run/hades/readonly/arangod.sock"
	defaultRWListenSocket = "/run/hades/readwrite/arangod.sock"
    // RO is group-connect; RW defaults to group-connect for dev (set 0600 in prod if desired).
    roSocketPermissions   = 0o660
	rwSocketPermissions   = 0o660
)

var cursorPathRegexp = regexp.MustCompile(`^(/_db/[^/]+)?/_api/cursor(?:/[0-9]+)?$`)

var hopByHopHeaders = []string{
	"Connection",
	"Proxy-Connection",
	"Keep-Alive",
	"TE",
	"Trailer",
	"Transfer-Encoding",
	"Upgrade",
}

type BodyPeeker func(limit int64) ([]byte, error)

// UnixReverseProxy forwards HTTP requests to an upstream exposed via Unix socket.
type UnixReverseProxy struct {
	upstreamSocket string
	allowFunc      func(*http.Request, BodyPeeker) error
	client         *http.Client
}

func newUnixReverseProxy(upstreamSocket string, allowFunc func(*http.Request, BodyPeeker) error) *UnixReverseProxy {
	transport := newUnixTransport(upstreamSocket)
	timeoutSec := getEnv("PROXY_CLIENT_TIMEOUT_SECONDS", getEnv("CLIENT_TIMEOUT_SECONDS", "120"))
	timeout := 120 * time.Second
	if d, err := time.ParseDuration(timeoutSec + "s"); err == nil {
		timeout = d
	}
	if timeoutSec == "0" {
		log.Printf("proxy client timeout: disabled (0s)")
		return &UnixReverseProxy{
			upstreamSocket: upstreamSocket,
			allowFunc:      allowFunc,
			client: &http.Client{
				Transport: transport,
				Timeout:   0,
			},
		}
	}
	log.Printf("proxy client timeout: %s", timeout)
	return &UnixReverseProxy{
		upstreamSocket: upstreamSocket,
		allowFunc:      allowFunc,
		client: &http.Client{
			Transport: transport,
			Timeout:   timeout,
		},
	}
}

func newUnixTransport(socketPath string) *http.Transport {
	dialTimeoutSec := getEnv("PROXY_DIAL_TIMEOUT_SECONDS", "10")
	dialTimeout := 10 * time.Second
	if d, err := time.ParseDuration(dialTimeoutSec + "s"); err == nil {
		dialTimeout = d
	}
	dialer := &net.Dialer{Timeout: dialTimeout}
	transport := &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			return dialer.DialContext(ctx, "unix", socketPath)
		},
		ForceAttemptHTTP2: true,
	}
	if err := http2.ConfigureTransport(transport); err != nil {
		log.Fatalf("failed to configure HTTP/2 transport: %v", err)
	}
	return transport
}

func (p *UnixReverseProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var cachedBody []byte
	bodyConsumed := false

	bodyReader := func(limit int64) ([]byte, error) {
		if bodyConsumed {
			return cachedBody, nil
		}
		if r.Body == nil {
			bodyConsumed = true
			return nil, nil
		}
		defer func() {
			bodyConsumed = true
		}()

		var buf bytes.Buffer
		if limit > 0 {
			lr := &io.LimitedReader{R: r.Body, N: limit + 1}
			if _, err := buf.ReadFrom(lr); err != nil {
				_ = r.Body.Close()
				return nil, err
			}
			if lr.N <= 0 {
				_ = r.Body.Close()
				return nil, fmt.Errorf("request body exceeds inspection limit (%d bytes)", limit)
			}
		} else {
			if _, err := buf.ReadFrom(r.Body); err != nil {
				_ = r.Body.Close()
				return nil, err
			}
		}

		if err := r.Body.Close(); err != nil {
			log.Printf("warning: failed to close request body: %v", err)
		}
		cachedBody = append([]byte(nil), buf.Bytes()...)
		return cachedBody, nil
	}

	if err := p.allowFunc(r, bodyReader); err != nil {
		http.Error(w, err.Error(), http.StatusForbidden)
		return
	}

	var upstreamBody io.ReadCloser
	if bodyConsumed {
		upstreamBody = io.NopCloser(bytes.NewReader(cachedBody))
	} else {
		upstreamBody = r.Body
	}

	upstreamURL := buildUpstreamURL(r)
	upstreamReq, err := http.NewRequestWithContext(r.Context(), r.Method, upstreamURL, upstreamBody)
	if err != nil {
		http.Error(w, "failed to build upstream request", http.StatusInternalServerError)
		return
	}

	copyHeaders(upstreamReq.Header, r.Header)
	if bodyConsumed {
		upstreamReq.ContentLength = int64(len(cachedBody))
	}

	resp, err := p.client.Do(upstreamReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("upstream error: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	copyHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	if _, err := io.Copy(w, resp.Body); err != nil {
		log.Printf("warning: failed to copy upstream response: %v", err)
	}
}

func copyHeaders(dst, src http.Header) {
	cleaned := cloneHeader(src)
	stripHopHeaders(cleaned)

	for key := range dst {
		dst.Del(key)
	}

	for key, values := range cleaned {
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func stripHopHeaders(header http.Header) {
	connectionValues := header.Values("Connection")
	for _, value := range connectionValues {
		for _, token := range strings.Split(value, ",") {
			headerName := strings.TrimSpace(token)
			if headerName == "" {
				continue
			}
			header.Del(http.CanonicalHeaderKey(headerName))
		}
	}

	for _, hopHeader := range hopByHopHeaders {
		header.Del(hopHeader)
	}
}

func cloneHeader(src http.Header) http.Header {
	cloned := make(http.Header, len(src))
	for key, values := range src {
		copied := make([]string, len(values))
		copy(copied, values)
		cloned[key] = copied
	}
	return cloned
}

func buildUpstreamURL(r *http.Request) string {
	var builder strings.Builder
	builder.WriteString("http://arangodb")
	builder.WriteString(r.URL.Path)
	if raw := r.URL.RawQuery; raw != "" {
		builder.WriteByte('?')
		builder.WriteString(raw)
	}
	return builder.String()
}

func removeIfExists(path string) {
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		log.Fatalf("failed to remove existing socket %s: %v", path, err)
	}
}

func ensureParentDir(path string) error {
	dir := filepath.Dir(path)
	if dir == "." || dir == "/" {
		return nil
	}
	return os.MkdirAll(dir, 0o750)
}

func ensureSocketMode(path string, mode os.FileMode) {
	if err := os.Chmod(path, mode); err != nil {
		log.Fatalf("failed to chmod %s: %v", path, err)
	}
}

func getEnv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func logRequests(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		loggedPath := r.URL.Path
		if r.URL.RawQuery != "" {
			loggedPath += "?<redacted>"
		}
		log.Printf("%s %s", r.Method, loggedPath)
		handler.ServeHTTP(w, r)
	})
}

func isCursorPath(path string) bool {
	return cursorPathRegexp.MatchString(path)
}
