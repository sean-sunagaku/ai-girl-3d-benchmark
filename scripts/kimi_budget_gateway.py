"""Temporary credential relay and conservative USD 5 guard for this run.

The official API key arrives over a user-only local socket and stays in memory.
OpenCode receives only a dummy local token. No request bodies/keys are logged.
"""
import hashlib
import http.server
import json
import os
from pathlib import Path
import socket
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(os.environ.get('KIMI_RUN_CONTROL_DIR', '.work/kimi-control')).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
SOCKET = str(ROOT / 'credential.sock')
MODEL = 'kimi-k3.1-pro'
BASE = 'https://api.moonshot.ai/v1'
CAP = 5.0
spent = 0.0
records = []
lock = threading.Lock()
api_key = None

def usage_upper_cost(usage):
    """Fresh input includes both input and default cache-write costs.

    Cache-hit input is billed only at its published cache-hit rate. Reasoning
    is already part of completion_tokens and must not be counted twice.
    """
    prompt = int(usage.get('prompt_tokens',0))
    details=usage.get('prompt_tokens_details') or {}
    cached=int(details.get('cached_tokens',usage.get('cached_tokens',0)) or 0)
    if cached < 0 or cached > prompt:
        raise ValueError('Invalid cached token count')
    return (prompt-cached)*0.000004 + cached*0.0000002 + int(usage['completion_tokens'])*0.000010

def upstream(path, data=None):
    raw = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(BASE + path, data=raw, headers={
        'Authorization': 'Bearer ' + api_key,
        'Content-Type': 'application/json',
    })
    return urllib.request.urlopen(req, timeout=900)

def persist():
    (ROOT / 'usage.json').write_text(json.dumps({
        'model': MODEL, 'base_url': BASE, 'cap_usd': CAP,
        'conservative_committed_usd': round(spent, 8), 'requests': records,
        'billing_note': 'Upper bound: uncached input at USD 4/MTok (input + default cache write), cache hits at USD 0.20/MTok, output including reasoning at USD 10/MTok. Each prospective request reserves all input as uncached plus maximum output; uncertain completions retain full reservation. Provider billing is authoritative.'
    }, indent=2))

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == '/stats':
            return self.reply(200, json.loads((ROOT / 'usage.json').read_text()))
        if self.path == '/v1/models':
            return self.reply(200, {'data': [{'id': MODEL, 'object': 'model', 'owned_by': 'moonshotai'}]})
        return self.reply(404, {'error': {'message': 'Unsupported local route'}})

    def do_POST(self):
        global spent
        if self.path != '/v1/chat/completions':
            return self.reply(404, {'error': {'message': 'Only official chat completions are allowed'}})
        with lock:
            reservation = None
            record = {'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if body.get('model') != MODEL:
                    return self.reply(400, {'error': {'message': 'Model mismatch; no fallback permitted'}})
                # Preserve the actual prompts/tool history. Apply only harness limits.
                body['max_tokens'] = min(int(body.get('max_tokens', 32768)), 32768)
                if body.get('reasoning_effort') not in ('low','high','max'):
                    body['reasoning_effort'] = 'high'
                body.pop('temperature', None)
                body.pop('top_p', None)
                if body.get('stream'):
                    body['stream_options'] = {'include_usage': True}
                estimate_body = {k: body[k] for k in ('model','messages','tools') if k in body}
                with upstream('/tokenizers/estimate-token-count', estimate_body) as r:
                    estimate = json.load(r)['data']['total_tokens']
                reservation = (estimate + 1024) * 0.000004 + body['max_tokens'] * 0.000010
                if spent + reservation > CAP:
                    return self.reply(402, {'error': {'message': 'Local USD 5 budget guard stopped before another paid request'}})
                spent += reservation
                record.update(estimated_input_tokens=estimate, maximum_output_tokens=body['max_tokens'],
                              reservation_usd=round(reservation,8), model=MODEL,
                              request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest())
                records.append(record)
                persist()
                response = upstream('/chat/completions', body)
                self.send_response(response.status)
                self.send_header('Content-Type', response.headers.get('Content-Type','application/json'))
                self.send_header('Connection', 'close')
                self.end_headers()
                usage = None
                response_model = None
                if body.get('stream'):
                    for line in response:
                        self.wfile.write(line)
                        self.wfile.flush()
                        if line.startswith(b'data: '):
                            try:
                                event = json.loads(line[6:])
                                usage = event.get('usage') or usage
                                response_model = event.get('model') or response_model
                            except (ValueError, TypeError):
                                pass
                else:
                    raw = response.read()
                    event = json.loads(raw)
                    usage = event.get('usage')
                    response_model = event.get('model')
                    self.wfile.write(raw)
                response.close()
                if usage:
                    cost = usage_upper_cost(usage)
                    spent += cost - reservation
                    record.update(usage=usage, conservative_cost_usd=round(cost,8), response_model=response_model,
                                  status='complete')
                else:
                    record['status']='usage_missing_reservation_retained'
                persist()
            except urllib.error.HTTPError as e:
                if reservation is not None:
                    # Non-successful requests do not produce completion tokens.
                    spent -= reservation
                    record.update(status='upstream_error', http_status=e.code)
                    persist()
                try:
                    payload = json.loads(e.read().replace(api_key.encode(),b'[REDACTED]'))
                except Exception:
                    payload = {'error':{'message':'Official API request failed','status':e.code}}
                self.reply(e.code, payload)
            except Exception as e:
                if reservation is not None:
                    record.update(status='interrupted_reservation_retained',error_type=type(e).__name__)
                    persist()
                try:
                    self.reply(502, {'error': {'message': 'Local gateway failure: '+type(e).__name__}})
                except Exception:
                    pass

if __name__ == '__main__':
    os.umask(0o077)
    if (ROOT/'usage.json').is_file():
        previous=json.loads((ROOT/'usage.json').read_text())
        records=previous['requests']
        for record in records:
            if record.get('status')=='complete' and record.get('usage'):
                cost=usage_upper_cost(record['usage'])
                record['conservative_cost_usd']=round(cost,8)
                spent+=cost
            elif record.get('status')!='upstream_error':
                spent+=record['reservation_usd']
    if os.path.exists(SOCKET):
        os.unlink(SOCKET)
    listener = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    listener.bind(SOCKET)
    os.chmod(SOCKET,0o600)
    listener.listen(1)
    print('Waiting for approved credential on local user-only socket.',flush=True)
    conn,_ = listener.accept()
    raw = b''
    while b'\n' not in raw:
        chunk=conn.recv(4096)
        if not chunk:break
        raw+=chunk
    api_key = raw.decode().strip()
    raw = b''
    listener.close()
    os.unlink(SOCKET)
    try:
        with upstream('/models') as r:
            ids=[m['id'] for m in json.load(r)['data']]
        if MODEL not in ids:
            raise RuntimeError('Requested model is not in official account model list')
        with upstream('/users/me/balance') as r:
            balance=json.load(r)
        (ROOT/'preflight-api.json').write_text(json.dumps({'model_present':True,'model':MODEL,'balance':balance},indent=2))
        persist()
        server=http.server.HTTPServer(('127.0.0.1',0),Handler)
        (ROOT/'gateway-port.json').write_text(json.dumps({'port':server.server_port}))
        conn.sendall(b'Credential received in memory; exact model confirmed.\n')
        conn.close()
        print('Official model and balance verified. Local gateway on port '+str(server.server_port),flush=True)
        server.serve_forever()
    finally:
        api_key=None
