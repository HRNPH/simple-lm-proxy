FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /lm-prox .

FROM alpine:3.22
RUN apk add --no-cache ca-certificates
COPY --from=build /lm-prox /usr/local/bin/lm-prox
EXPOSE 10000
ENTRYPOINT ["lm-prox"]
