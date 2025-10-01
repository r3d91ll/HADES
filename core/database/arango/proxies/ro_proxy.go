package proxies

import (
    "encoding/json"
    "fmt"
    "log"
    "net"
    "net/http"
    "strings"
    "strconv"
)

func RunReadOnlyProxy() error {
	listenSocket := getEnv("LISTEN_SOCKET", defaultROListenSocket)
	upstreamSocket := getEnv("UPSTREAM_SOCKET", defaultUpstreamSocket)

	if err := ensureParentDir(listenSocket); err != nil {
		return fmt.Errorf("failed to prepare directory for %s: %w", listenSocket, err)
	}
	removeIfExists(listenSocket)

	proxy := newUnixReverseProxy(upstreamSocket, allowReadOnly)

	listener, err := net.Listen("unix", listenSocket)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", listenSocket, err)
	}
	ensureSocketMode(listenSocket, roSocketPermissions)

	server := &http.Server{Handler: logRequests(proxy)}

	log.Printf("Read-only proxy listening on %s -> %s", listenSocket, upstreamSocket)
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("proxy server error: %w", err)
	}
	return nil
}

func allowReadOnly(r *http.Request, peek BodyPeeker) error {
	switch r.Method {
	case http.MethodGet, http.MethodHead, http.MethodOptions:
		return nil
	case http.MethodPost:
		if isCursorPath(r.URL.Path) {
			// Enforce a configurable peek limit to avoid scanning arbitrarily large bodies.
			// Default 128 KiB; override via AQL_PEEK_LIMIT_BYTES.
			peekLimit := int64(128 * 1024)
			if v := getEnv("AQL_PEEK_LIMIT_BYTES", ""); v != "" {
				if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
					peekLimit = n
				}
			}
			body, err := peek(peekLimit)
			if err != nil {
				return err
			}
			var payload struct {
				Query string `json:"query"`
			}
			if err := json.Unmarshal(body, &payload); err == nil && payload.Query != "" {
				upper := strings.ToUpper(payload.Query)
				tokens := strings.FieldsFunc(upper, func(r rune) bool {
					return r < 'A' || r > 'Z'
				})
				forbiddenKeywords := map[string]struct{}{
					"INSERT":   {},
					"UPDATE":   {},
					"UPSERT":   {},
					"REMOVE":   {},
					"REPLACE":  {},
					"TRUNCATE": {},
					"DROP":     {},
					"CREATE":   {},
					"ALTER":    {},
					"RENAME":   {},
					"GRANT":    {},
					"REVOKE":   {},
					"GRAPH":    {},
				}
				for _, token := range tokens {
					if _, forbidden := forbiddenKeywords[token]; forbidden {
						return fmt.Errorf("forbidden keyword %q detected in AQL", token)
					}
				}
				return nil
			}
			// Fallback: conservative scan of raw body
			upper := strings.ToUpper(string(body))
			for _, keyword := range []string{"INSERT", "UPDATE", "UPSERT", "REMOVE", "REPLACE", "TRUNCATE", "DROP", "CREATE", "ALTER", "RENAME", "GRANT", "REVOKE", "GRAPH"} {
				if strings.Contains(upper, keyword) {
					return fmt.Errorf("forbidden keyword %q detected in request body", keyword)
				}
			}
			return nil
		}
	case http.MethodPut, http.MethodDelete:
		if isCursorPath(r.URL.Path) {
			return nil
		}
	}
	return fmt.Errorf("method %s not permitted on %s", r.Method, r.URL.Path)
}
