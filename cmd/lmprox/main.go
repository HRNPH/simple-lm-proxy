package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"lm-prox/internal/bedrock"
	"lm-prox/internal/server"
)

func main() {
	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = os.Getenv("AWS_DEFAULT_REGION")
	}
	if region == "" {
		region = bedrock.DefaultRegion
	}

	client, err := bedrock.New(region)
	if err != nil {
		log.Fatalf("load aws config: %v", err)
	}

	authMode := "aws-credential-chain"
	if os.Getenv("AWS_BEARER_TOKEN_BEDROCK") != "" {
		authMode = "bedrock-bearer-token"
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = server.DefaultPort
	}

	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           server.New(client).Handler(),
		ReadHeaderTimeout: 15 * time.Second,
	}

	go func() {
		log.Printf("lm-prox listening on :%s region=%s auth=%s", port, region, authMode)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server: %v", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	log.Println("shutting down")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx)
}
