"""Minimal TCP relay (2026-11-12, parallel arm B): containers reach the second d1 API at <docker bridge>:8702,
forwarded to 127.0.0.1:18702 (an SSH reverse tunnel). Usage: python3 tcp_relay.py BIND_HOST BIND_PORT DST_PORT"""
import socket, sys, threading
host, port, dst = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
def pipe(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d: break
            b.sendall(d)
    except OSError: pass
    finally:
        for s in (a, b):
            try: s.shutdown(socket.SHUT_RDWR)
            except OSError: pass
srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind((host, port)); srv.listen(64)
while True:
    c, _ = srv.accept()
    try: u = socket.create_connection(("127.0.0.1", dst))
    except OSError: c.close(); continue
    threading.Thread(target=pipe, args=(c, u), daemon=True).start(); threading.Thread(target=pipe, args=(u, c), daemon=True).start()
