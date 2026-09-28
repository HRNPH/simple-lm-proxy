package bedrock

import (
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

var imageClient = &http.Client{Timeout: 20 * time.Second}

const maxImageBytes = 4 << 20

func fetchImage(url string) (*types.ImageBlock, error) {
	var data []byte
	var mime string

	if strings.HasPrefix(url, "data:") {
		comma := strings.Index(url, ",")
		if comma < 0 {
			return nil, fmt.Errorf("invalid data URI")
		}
		meta := url[5:comma]
		payload := url[comma+1:]
		if !strings.Contains(meta, "base64") {
			return nil, fmt.Errorf("only base64 data URIs supported")
		}
		mime = strings.TrimSuffix(strings.Split(meta, ";")[0], ";")
		var err error
		data, err = base64.StdEncoding.DecodeString(payload)
		if err != nil {
			return nil, fmt.Errorf("invalid base64 image data: %w", err)
		}
	} else if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		req, err := http.NewRequest(http.MethodGet, url, nil)
		if err != nil {
			return nil, err
		}
		resp, err := imageClient.Do(req)
		if err != nil {
			return nil, fmt.Errorf("fetch image: %w", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("fetch image: status %d", resp.StatusCode)
		}
		mime = strings.TrimSpace(strings.Split(resp.Header.Get("Content-Type"), ";")[0])
		data, err = io.ReadAll(io.LimitReader(resp.Body, maxImageBytes+1))
		if err != nil {
			return nil, fmt.Errorf("read image: %w", err)
		}
	} else {
		return nil, fmt.Errorf("unsupported image URL scheme (use data: or http(s):)")
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("empty image data")
	}
	if len(data) > maxImageBytes {
		return nil, fmt.Errorf("image exceeds %d bytes", maxImageBytes)
	}

	format := detectImageFormat(data, mime)
	if format == "" {
		return nil, fmt.Errorf("unsupported image format (png, jpeg, gif, webp only)")
	}

	return &types.ImageBlock{
		Format: format,
		Source: &types.ImageSourceMemberBytes{Value: data},
	}, nil
}

func detectImageFormat(data []byte, mime string) types.ImageFormat {
	switch {
	case len(data) >= 8 && data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4e && data[3] == 0x47:
		return types.ImageFormatPng
	case len(data) >= 3 && data[0] == 0xff && data[1] == 0xd8 && data[2] == 0xff:
		return types.ImageFormatJpeg
	case len(data) >= 6 && (string(data[:6]) == "GIF87a" || string(data[:6]) == "GIF89a"):
		return types.ImageFormatGif
	case len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP":
		return types.ImageFormatWebp
	}

	switch strings.ToLower(mime) {
	case "image/png":
		return types.ImageFormatPng
	case "image/jpeg", "image/jpg":
		return types.ImageFormatJpeg
	case "image/gif":
		return types.ImageFormatGif
	case "image/webp":
		return types.ImageFormatWebp
	}
	return ""
}
