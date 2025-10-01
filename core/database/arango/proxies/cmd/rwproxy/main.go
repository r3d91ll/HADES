package main

import (
	"log"

	"github.com/r3d91ll/HADES-Lab/core/database/arango/proxies"
)

func main() {
	if err := proxies.RunReadWriteProxy(); err != nil {
		log.Fatal(err)
	}
}
