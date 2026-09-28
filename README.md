# lm prox

A small Go server that speaks the OpenAI Chat Completions API and forwards requests to Amazon Bedrock.

## Features

- OpenAI style `POST /v1/chat/completions` with streaming, tool calls, and images
- `GET /v1/models` lists every supported model alias
- `GET /health` returns a simple status check
- Auth via the AWS credential chain or a Bedrock bearer token
- Short aliases for Claude, GPT, Grok, Kimi, and Nova models (see `modelAliases` in `main.go`)

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `10000` | HTTP listen port |
| `AWS_REGION` | `us-east-1` | Bedrock region |
| `AWS_PROFILE` | none | AWS shared config profile |
| `AWS_BEARER_TOKEN_BEDROCK` | none | Bedrock bearer token, used when set |

Standard AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`) are read from the environment or from `~/.aws`.

## Run

From source:

```
go run .
```

With Docker:

```
docker compose up --build
```

## Usage

```
curl http://localhost:10000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"sonnet-4.6","messages":[{"role":"user","content":"hello"}]}'
```

Set `stream: true` for server sent events.

## License

MIT
