# Email Service - Integration Guide

> For developers integrating the email service into their applications

## Overview

The Email Service provides a simple HTTP API for sending transactional emails (OTP codes, notifications, alerts). It's a shared service available to all internal applications.

## Quick Start

```bash
curl -X POST https://email-api.internal.yourcompany.com/send \
  -H "x-api-key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "to": "user@example.com",
    "subject": "Your verification code",
    "body_text": "Your code is: 123456"
  }'
```

---

## Configuration

Add these environment variables to your application:

| Variable | Required | Description |
|----------|----------|-------------|
| `EMAIL_SERVICE_URL` | Yes | API endpoint URL |
| `EMAIL_SERVICE_API_KEY` | Yes | Your API key |
| `AUTH_DEV_MODE` | No | Set `true` to skip API calls locally |

**Example `.env`:**
```env
EMAIL_SERVICE_URL=https://xxxxxxxxxx.execute-api.ap-southeast-2.amazonaws.com/prod/send
EMAIL_SERVICE_API_KEY=aBcDeFgHiJkLmNoPqRsTuVwXyZ
AUTH_DEV_MODE=false
```

---

## API Reference

### Send Email

```
POST /send
```

**Headers:**
| Header | Required | Value |
|--------|----------|-------|
| `x-api-key` | Yes | Your API key |
| `Content-Type` | Yes | `application/json` |

**Request Body:**
```json
{
  "to": "user@example.com",
  "subject": "Your subject line",
  "body_text": "Plain text content",
  "body_html": "<p>HTML content</p>"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `to` | string | Yes | Recipient email |
| `subject` | string | Yes | Subject line (max 200 chars) |
| `body_text` | string | Yes | Plain text body |
| `body_html` | string | No | HTML body (optional) |

**Success Response (200):**
```json
{
  "success": true,
  "message_id": "0100018d1234abcd-..."
}
```

**Error Response (400/500):**
```json
{
  "success": false,
  "error": "Missing required field: subject",
  "error_code": "VALIDATION_ERROR"
}
```

---

## Code Examples

### Python

```python
import os
import httpx

class EmailService:
    def __init__(self):
        self.url = os.environ["EMAIL_SERVICE_URL"]
        self.api_key = os.environ["EMAIL_SERVICE_API_KEY"]
        self.dev_mode = os.getenv("AUTH_DEV_MODE", "").lower() == "true"

    def send(self, to: str, subject: str, body_text: str, body_html: str = None) -> bool:
        """Send an email. Returns True on success."""

        if self.dev_mode:
            print(f"[DEV EMAIL] To: {to} | Subject: {subject}")
            print(f"[DEV EMAIL] Body: {body_text[:100]}...")
            return True

        payload = {
            "to": to,
            "subject": subject,
            "body_text": body_text
        }
        if body_html:
            payload["body_html"] = body_html

        response = httpx.post(
            self.url,
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=10.0
        )

        if response.status_code == 200:
            return response.json().get("success", False)

        # Log error for debugging
        print(f"Email failed: {response.status_code} - {response.text}")
        return False


# Usage
email = EmailService()
email.send(
    to="user@example.com",
    subject="Your login code",
    body_text="Your code is: 123456"
)
```

### Python (OTP Helper)

```python
import os
import random
import time
from dataclasses import dataclass

@dataclass
class OTPRecord:
    code: str
    expires_at: float

class OTPService:
    """Generate and validate OTP codes with email delivery."""

    def __init__(self, email_service: EmailService):
        self.email = email_service
        self.codes: dict[str, OTPRecord] = {}  # email -> OTPRecord
        self.expiry_seconds = 600  # 10 minutes

    def send_code(self, email: str) -> bool:
        """Generate OTP and send via email."""
        code = f"{random.randint(0, 999999):06d}"
        expires_at = time.time() + self.expiry_seconds

        self.codes[email] = OTPRecord(code=code, expires_at=expires_at)

        return self.email.send(
            to=email,
            subject="Your verification code",
            body_text=f"Your verification code is: {code}\n\nThis code expires in 10 minutes.",
            body_html=f"""
                <h2>Your verification code</h2>
                <p style="font-size: 32px; font-weight: bold; letter-spacing: 8px;">{code}</p>
                <p>This code expires in 10 minutes.</p>
            """
        )

    def verify_code(self, email: str, code: str) -> bool:
        """Verify OTP code. Returns True if valid."""
        record = self.codes.get(email)

        if not record:
            return False

        if time.time() > record.expires_at:
            del self.codes[email]
            return False

        if record.code != code:
            return False

        # Valid - remove used code
        del self.codes[email]
        return True
```

### Node.js / TypeScript

```typescript
import axios from 'axios';

interface EmailPayload {
  to: string;
  subject: string;
  body_text: string;
  body_html?: string;
}

interface EmailResponse {
  success: boolean;
  message_id?: string;
  error?: string;
  error_code?: string;
}

class EmailService {
  private url: string;
  private apiKey: string;
  private devMode: boolean;

  constructor() {
    this.url = process.env.EMAIL_SERVICE_URL!;
    this.apiKey = process.env.EMAIL_SERVICE_API_KEY!;
    this.devMode = process.env.AUTH_DEV_MODE === 'true';
  }

  async send(payload: EmailPayload): Promise<boolean> {
    if (this.devMode) {
      console.log(`[DEV EMAIL] To: ${payload.to} | Subject: ${payload.subject}`);
      return true;
    }

    try {
      const response = await axios.post<EmailResponse>(this.url, payload, {
        headers: {
          'x-api-key': this.apiKey,
          'Content-Type': 'application/json'
        },
        timeout: 10000
      });

      return response.data.success;
    } catch (error) {
      console.error('Email failed:', error);
      return false;
    }
  }
}

// Usage
const email = new EmailService();
await email.send({
  to: 'user@example.com',
  subject: 'Your login code',
  body_text: 'Your code is: 123456'
});
```

### Go

```go
package email

import (
    "bytes"
    "encoding/json"
    "net/http"
    "os"
    "time"
)

type EmailService struct {
    URL     string
    APIKey  string
    DevMode bool
    Client  *http.Client
}

type EmailPayload struct {
    To       string `json:"to"`
    Subject  string `json:"subject"`
    BodyText string `json:"body_text"`
    BodyHTML string `json:"body_html,omitempty"`
}

type EmailResponse struct {
    Success   bool   `json:"success"`
    MessageID string `json:"message_id,omitempty"`
    Error     string `json:"error,omitempty"`
}

func NewEmailService() *EmailService {
    return &EmailService{
        URL:     os.Getenv("EMAIL_SERVICE_URL"),
        APIKey:  os.Getenv("EMAIL_SERVICE_API_KEY"),
        DevMode: os.Getenv("AUTH_DEV_MODE") == "true",
        Client:  &http.Client{Timeout: 10 * time.Second},
    }
}

func (s *EmailService) Send(payload EmailPayload) (bool, error) {
    if s.DevMode {
        println("[DEV EMAIL] To:", payload.To, "| Subject:", payload.Subject)
        return true, nil
    }

    body, _ := json.Marshal(payload)
    req, _ := http.NewRequest("POST", s.URL, bytes.NewBuffer(body))
    req.Header.Set("x-api-key", s.APIKey)
    req.Header.Set("Content-Type", "application/json")

    resp, err := s.Client.Do(req)
    if err != nil {
        return false, err
    }
    defer resp.Body.Close()

    var result EmailResponse
    json.NewDecoder(resp.Body).Decode(&result)
    return result.Success, nil
}
```

---

## Local Development

Set `AUTH_DEV_MODE=true` in your local `.env` to skip actual email sending:

```env
AUTH_DEV_MODE=true
EMAIL_SERVICE_URL=https://example.com  # not used in dev mode
EMAIL_SERVICE_API_KEY=not-used         # not used in dev mode
```

In dev mode, emails are logged to console instead of being sent. This allows:
- Fast local testing without AWS credentials
- No email spam during development
- Visible OTP codes in terminal

---

## Rate Limits

| Limit | Value |
|-------|-------|
| Requests per second | 50 |
| Burst | 100 |
| Daily quota | 10,000 |

If you exceed rate limits, you'll receive a `429 Too Many Requests` response.

---

## Error Handling

Always check the response and handle failures gracefully:

```python
def send_with_retry(email_service, to, subject, body, max_retries=3):
    """Send email with retry logic."""
    for attempt in range(max_retries):
        try:
            if email_service.send(to, subject, body):
                return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff

    return False
```

**Common errors:**

| Error Code | Cause | Solution |
|------------|-------|----------|
| `VALIDATION_ERROR` | Missing/invalid field | Check request body |
| `INVALID_EMAIL` | Bad email format | Validate email before sending |
| `MESSAGE_REJECTED` | SES rejected email | Check recipient is valid |
| `429` | Rate limited | Implement backoff/retry |

---

## Best Practices

1. **Validate emails client-side** before calling the API
2. **Use HTML templates** for consistent branding
3. **Keep subject lines short** (< 50 chars for mobile)
4. **Include plain text** for email clients that don't support HTML
5. **Don't send sensitive data** in subject lines
6. **Implement retry logic** for transient failures
7. **Use dev mode locally** to avoid email spam

---

## Getting an API Key

Contact the platform team to request an API key for your service. Provide:
- Service name
- Expected daily volume
- Contact person

---

## Support

- **Issues**: Contact platform team
- **Service status**: Check CloudWatch dashboard
