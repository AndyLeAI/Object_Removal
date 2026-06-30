from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import mimetypes
import webbrowser

PORT = 5177
ROOT = Path(__file__).resolve().parent
mimetypes.add_type('application/wasm', '.wasm')

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

if __name__ == '__main__':
    url = f'http://localhost:{PORT}'
    print(f'Obj Remover running at {url}')
    print('Keep this window open while using the app.')
    try:
        webbrowser.open(url)
    except Exception:
        pass
    ThreadingHTTPServer(('localhost', PORT), Handler).serve_forever()
