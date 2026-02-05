# Email Lambda Service - Build Specification

> For the dev team building the Lambda function

## Overview

Build a reusable AWS Lambda function that sends emails via SES, exposed through API Gateway with API key authentication.

## Architecture

```
Client → API Gateway (x-api-key) → Lambda → SES → Email
```

## Requirements

### Functional
- Accept email requests via HTTP POST
- Send emails using AWS SES
- Support plain text and HTML body
- Return success/failure response with message ID

### Non-Functional
- Response time < 3 seconds
- 99.9% availability
- Rate limit: 50 requests/second
- Daily quota: 10,000 emails

---

## API Specification

### Endpoint

```
POST /send
Host: {api-gateway-url}
Content-Type: application/json
x-api-key: {api-key}
```

### Request Body

```json
{
  "to": "user@example.com",
  "subject": "Your login code",
  "body_text": "Your code is: 123456",
  "body_html": "<p>Your code is: <strong>123456</strong></p>"
}
```

| Field | Type | Required | Max Length | Description |
|-------|------|----------|------------|-------------|
| `to` | string | Yes | 254 | Recipient email address |
| `subject` | string | Yes | 200 | Email subject line |
| `body_text` | string | Yes | 10,000 | Plain text body |
| `body_html` | string | No | 50,000 | HTML body (optional) |

### Response - Success (200)

```json
{
  "success": true,
  "message_id": "0100018d1234abcd-12345678-1234-1234-1234-123456789abc-000000"
}
```

### Response - Validation Error (400)

```json
{
  "success": false,
  "error": "Missing required field: subject",
  "error_code": "VALIDATION_ERROR"
}
```

### Response - SES Error (400)

```json
{
  "success": false,
  "error": "Email address is not verified",
  "error_code": "MESSAGE_REJECTED"
}
```

### Response - Server Error (500)

```json
{
  "success": false,
  "error": "Internal server error",
  "error_code": "INTERNAL_ERROR"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Missing or invalid request field |
| `INVALID_EMAIL` | 400 | Invalid email address format |
| `MESSAGE_REJECTED` | 400 | SES rejected the email |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

## Lambda Implementation

### Configuration

| Property | Value |
|----------|-------|
| Runtime | Python 3.11+ |
| Handler | `handler.send_email` |
| Timeout | 10 seconds |
| Memory | 128 MB |

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SES_FROM_EMAIL` | Verified sender email | `noreply@yourdomain.com` |
| `AWS_REGION` | AWS region | `ap-southeast-2` |

### IAM Role Permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SendEmail",
      "Effect": "Allow",
      "Action": [
        "ses:SendEmail",
        "ses:SendRawEmail"
      ],
      "Resource": "*"
    },
    {
      "Sid": "Logging",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

### Reference Implementation

```python
import json
import os
import re
import boto3
from botocore.exceptions import ClientError

ses = boto3.client('ses')
FROM_EMAIL = os.environ['SES_FROM_EMAIL']
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')


def send_email(event, context):
    """Send email via AWS SES."""
    try:
        # Parse body if coming from API Gateway
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event

        # Validate required fields
        errors = validate_request(body)
        if errors:
            return response(400, {
                'success': False,
                'error': errors[0],
                'error_code': 'VALIDATION_ERROR'
            })

        # Validate email format
        if not EMAIL_REGEX.match(body['to']):
            return response(400, {
                'success': False,
                'error': 'Invalid email address format',
                'error_code': 'INVALID_EMAIL'
            })

        # Build email body
        email_body = {
            'Text': {'Data': body['body_text'], 'Charset': 'UTF-8'}
        }
        if body.get('body_html'):
            email_body['Html'] = {'Data': body['body_html'], 'Charset': 'UTF-8'}

        # Send via SES
        result = ses.send_email(
            Source=FROM_EMAIL,
            Destination={'ToAddresses': [body['to']]},
            Message={
                'Subject': {'Data': body['subject'], 'Charset': 'UTF-8'},
                'Body': email_body
            }
        )

        return response(200, {
            'success': True,
            'message_id': result['MessageId']
        })

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        return response(400, {
            'success': False,
            'error': error_message,
            'error_code': error_code.upper().replace(' ', '_')
        })

    except json.JSONDecodeError:
        return response(400, {
            'success': False,
            'error': 'Invalid JSON in request body',
            'error_code': 'VALIDATION_ERROR'
        })

    except Exception as e:
        # Log error for debugging (CloudWatch)
        print(f'Unexpected error: {str(e)}')
        return response(500, {
            'success': False,
            'error': 'Internal server error',
            'error_code': 'INTERNAL_ERROR'
        })


def validate_request(body: dict) -> list[str]:
    """Validate request body. Returns list of error messages."""
    errors = []
    required = ['to', 'subject', 'body_text']

    for field in required:
        if not body.get(field):
            errors.append(f'Missing required field: {field}')

    if body.get('subject') and len(body['subject']) > 200:
        errors.append('Subject must be 200 characters or less')

    if body.get('body_text') and len(body['body_text']) > 10000:
        errors.append('body_text must be 10,000 characters or less')

    if body.get('body_html') and len(body['body_html']) > 50000:
        errors.append('body_html must be 50,000 characters or less')

    return errors


def response(status_code: int, body: dict) -> dict:
    """Build API Gateway response."""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps(body)
    }
```

---

## API Gateway Configuration

### Setup

| Property | Value |
|----------|-------|
| Type | REST API |
| Endpoint | Regional |
| Stage | `prod` |

### Resource & Method

| Resource | Method | Auth | Integration |
|----------|--------|------|-------------|
| `/send` | POST | API Key Required | Lambda Proxy |

### Usage Plan

| Setting | Value |
|---------|-------|
| Throttle - Rate | 50 requests/second |
| Throttle - Burst | 100 requests |
| Quota | 10,000 requests/day |

---

## Deployment

### SAM Template

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Parameters:
  SesFromEmail:
    Type: String
    Description: Verified SES sender email
  Stage:
    Type: String
    Default: prod

Resources:
  EmailFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: email-service
      Runtime: python3.11
      Handler: handler.send_email
      Timeout: 10
      MemorySize: 128
      Environment:
        Variables:
          SES_FROM_EMAIL: !Ref SesFromEmail
      Policies:
        - Version: '2012-10-17'
          Statement:
            - Effect: Allow
              Action:
                - ses:SendEmail
                - ses:SendRawEmail
              Resource: '*'
      Events:
        SendEmail:
          Type: Api
          Properties:
            RestApiId: !Ref EmailApi
            Path: /send
            Method: POST

  EmailApi:
    Type: AWS::Serverless::Api
    Properties:
      StageName: !Ref Stage
      Auth:
        ApiKeyRequired: true

  ApiKey:
    Type: AWS::ApiGateway::ApiKey
    DependsOn: EmailApiStage
    Properties:
      Name: email-service-key
      Enabled: true
      StageKeys:
        - RestApiId: !Ref EmailApi
          StageName: !Ref Stage

  UsagePlan:
    Type: AWS::ApiGateway::UsagePlan
    DependsOn: EmailApiStage
    Properties:
      UsagePlanName: email-service-plan
      ApiStages:
        - ApiId: !Ref EmailApi
          Stage: !Ref Stage
      Throttle:
        BurstLimit: 100
        RateLimit: 50
      Quota:
        Limit: 10000
        Period: DAY

  UsagePlanKey:
    Type: AWS::ApiGateway::UsagePlanKey
    Properties:
      KeyId: !Ref ApiKey
      KeyType: API_KEY
      UsagePlanId: !Ref UsagePlan

Outputs:
  Endpoint:
    Value: !Sub 'https://${EmailApi}.execute-api.${AWS::Region}.amazonaws.com/${Stage}/send'
  ApiKeyId:
    Value: !Ref ApiKey
    Description: Run 'aws apigateway get-api-key --api-key <id> --include-value' to get value
```

### Deploy Commands

```bash
# Deploy
sam build
sam deploy --parameter-overrides SesFromEmail=noreply@yourdomain.com

# Get API key value
aws apigateway get-api-key --api-key <ApiKeyId> --include-value --query 'value' --output text
```

---

## SES Prerequisites

1. **Verify sender domain** in SES console
2. **Configure DKIM** for deliverability
3. **Request production access** (sandbox limits to verified recipients only)

---

## Testing

### Local with SAM

```bash
sam local start-api
curl -X POST http://localhost:3000/send \
  -H "Content-Type: application/json" \
  -d '{"to":"test@example.com","subject":"Test","body_text":"Hello"}'
```

### Deployed

```bash
curl -X POST https://{api-id}.execute-api.{region}.amazonaws.com/prod/send \
  -H "x-api-key: {your-api-key}" \
  -H "Content-Type: application/json" \
  -d '{"to":"test@example.com","subject":"Test","body_text":"Hello"}'
```

---

## Monitoring

- **CloudWatch Logs**: All invocations logged
- **CloudWatch Metrics**: Invocations, errors, duration
- **Recommended Alarm**: Error rate > 5% for 5 minutes

---

## Security Checklist

- [ ] SES sender domain verified with DKIM
- [ ] Lambda has minimum required permissions
- [ ] API key stored securely (not in code)
- [ ] Usage plan rate limiting configured
- [ ] CloudWatch logging enabled
- [ ] No PII logged (email addresses, content)
