# infra/ — System Services and Runtime Config

Parent: `../README.md`

Children
- systemd/ — service units for RO/RW proxies
- tmpfiles.d/ — runtime directory creation

Files
- `systemd/hades-roproxy.service` — RO proxy (group hades, socket 0660)
- `systemd/hades-rwproxy.service` — RW proxy (group hades, socket 0660 dev / 0600 prod)
- `tmpfiles.d/hades-proxy.conf` — creates `/run/hades/{readonly,readwrite}` at boot

Deploy
```
sudo install -m 0755 core/database/arango/proxies/bin/roproxy /usr/local/bin/hades-roproxy
sudo install -m 0755 core/database/arango/proxies/bin/rwproxy /usr/local/bin/hades-rwproxy
sudo install -m 0644 infra/systemd/hades-roproxy.service /etc/systemd/system/
sudo install -m 0644 infra/systemd/hades-rwproxy.service /etc/systemd/system/
sudo install -m 0644 infra/systemd/tmpfiles.d/hades-proxy.conf /etc/tmpfiles.d/
sudo systemd-tmpfiles --create && sudo systemctl daemon-reload
sudo systemctl enable --now hades-roproxy hades-rwproxy
```

Verify
```
ls -al /run/hades/readonly/arangod.sock  # srw-rw---- arangodb:hades
ls -al /run/hades/readwrite/arangod.sock # srw-rw---- arangodb:hades (or 0600 prod)
```
