package server

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/aws/smithy-go"
)

func mapError(err error) (int, string, string) {
	var ae smithy.APIError
	if errors.As(err, &ae) {
		msg := ae.ErrorMessage()
		switch ae.ErrorCode() {
		case "ValidationException":
			return http.StatusBadRequest, msg, "invalid_request_error"
		case "AccessDeniedException", "AccessDenied":
			return http.StatusForbidden, msg, "permission_error"
		case "UnauthorizedException", "UnrecognizedClientException", "InvalidSignatureException":
			return http.StatusUnauthorized, msg, "authentication_error"
		case "ThrottlingException", "Throttling", "TooManyRequestsException", "ServiceQuotaExceededException":
			return http.StatusTooManyRequests, msg, "rate_limit_error"
		case "ResourceNotFoundException":
			return http.StatusNotFound, msg, "invalid_request_error"
		case "ModelNotReadyException":
			return http.StatusServiceUnavailable, msg, "server_error"
		case "ServiceUnavailableException", "InternalServerException":
			return http.StatusInternalServerError, msg, "server_error"
		case "ModelTimeoutException":
			return http.StatusGatewayTimeout, msg, "server_error"
		default:
			return http.StatusInternalServerError, ae.ErrorCode() + ": " + msg, "api_error"
		}
	}

	msg := err.Error()
	lower := strings.ToLower(msg)
	if strings.Contains(lower, "sso session") ||
		strings.Contains(lower, "sso login") ||
		(strings.Contains(lower, "credential") && strings.Contains(lower, "expired")) ||
		strings.Contains(lower, "failed to get shared config profile") ||
		strings.Contains(lower, "expired or is otherwise invalid") {
		profile := os.Getenv("AWS_PROFILE")
		hint := "run: aws sso login"
		if profile != "" {
			hint = fmt.Sprintf("run: aws sso login --profile %s", profile)
		}
		return http.StatusUnauthorized,
			fmt.Sprintf("AWS credentials unavailable: %s (or set AWS_BEARER_TOKEN_BEDROCK)", hint),
			"authentication_error"
	}
	if strings.Contains(lower, "failed to retrieve") && strings.Contains(lower, "token") {
		return http.StatusUnauthorized,
			"AWS bearer token invalid - regenerate Bedrock API key and set AWS_BEARER_TOKEN_BEDROCK",
			"authentication_error"
	}
	return http.StatusInternalServerError, msg, "api_error"
}
