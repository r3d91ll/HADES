package proxies

import (
    "fmt"
    "log"
    "net"
    "net/http"
    "regexp"
)

func RunReadWriteProxy() error {
	listenSocket := getEnv("LISTEN_SOCKET", defaultRWListenSocket)
	upstreamSocket := getEnv("UPSTREAM_SOCKET", defaultUpstreamSocket)

	if err := ensureParentDir(listenSocket); err != nil {
		return fmt.Errorf("failed to prepare directory for %s: %w", listenSocket, err)
	}
	removeIfExists(listenSocket)

	proxy := newUnixReverseProxy(upstreamSocket, allowReadWrite)

	listener, err := net.Listen("unix", listenSocket)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", listenSocket, err)
	}
	ensureSocketMode(listenSocket, rwSocketPermissions)

	server := &http.Server{Handler: logRequests(proxy)}

	log.Printf("Read-write proxy listening on %s -> %s", listenSocket, upstreamSocket)
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("proxy server error: %w", err)
	}
	return nil
}

var dbScopedRW = regexp.MustCompile(`^/_db/[^/]+/_api/(document|index|collection|import|cursor)(/|$)`) // DB-scoped only

func allowReadWrite(r *http.Request, peek BodyPeeker) error {
    // Permit safe RO requests as-is
    if err := allowReadOnly(r, peek); err == nil {
        return nil
    }

    path := r.URL.Path
    if !dbScopedRW.MatchString(path) {
        // Explicitly forbid admin endpoints like /_api/database, /_api/view, /_api/analyzer
        return fmt.Errorf("path not permitted: %s (DB-scoped endpoints only)", path)
    }

    switch r.Method {
    case http.MethodPost:
        // POST is allowed for cursor/import/document/collection/index under DB scope
        return nil
    case http.MethodPut, http.MethodPatch, http.MethodDelete:
        // Mutations to document/collection/index are allowed; cursor DELETE allowed for cursor close
        return nil
    case http.MethodGet, http.MethodHead, http.MethodOptions:
        // Reads under DB scope are fine as well
        return nil
    }
    return fmt.Errorf("method %s not permitted on %s", r.Method, path)
}
