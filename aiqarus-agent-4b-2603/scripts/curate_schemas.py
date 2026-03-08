#!/usr/bin/env python3
"""
Enterprise Tool Schema Library Generator

Generates 400+ realistic enterprise tool-calling schemas across 24 domains.
Outputs: training schemas + held-out schemas for testing.

Design: Each domain defines entities, operations (by risk level), and
domain-specific parameter pools. The generator combines these to produce
unique, realistic schemas with proper descriptions, parameter types,
enums, and required fields.
"""

import json
import random
import re
from pathlib import Path
from collections import defaultdict


def pluralize(word: str) -> str:
    """Simple English pluralization for tool names."""
    if word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    return word + "s"

random.seed(42)

# ---------------------------------------------------------------------------
# Common parameter fragments reused across domains
# ---------------------------------------------------------------------------
COMMON_PARAMS = {
    "limit": {"type": "integer", "description": "Maximum number of results to return (default 25, max 100)"},
    "offset": {"type": "integer", "description": "Number of results to skip for pagination"},
    "sort_by": {"type": "string", "description": "Field name to sort results by"},
    "sort_order": {"type": "string", "enum": ["asc", "desc"], "description": "Sort direction"},
    "date_from": {"type": "string", "description": "Start date filter in ISO 8601 format (YYYY-MM-DD)"},
    "date_to": {"type": "string", "description": "End date filter in ISO 8601 format (YYYY-MM-DD)"},
    "include_archived": {"type": "boolean", "description": "Whether to include archived records"},
    "fields": {"type": "array", "items": {"type": "string"}, "description": "Specific fields to return in the response"},
    "reason": {"type": "string", "description": "Reason for this action (required for audit trail)"},
    "dry_run": {"type": "boolean", "description": "If true, validate the operation without executing it"},
    "notify": {"type": "boolean", "description": "Whether to send notifications about this action"},
    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags to associate with this record"},
    "notes": {"type": "string", "description": "Free-text notes or comments"},
}

# ---------------------------------------------------------------------------
# 24 Enterprise Domains
# ---------------------------------------------------------------------------
DOMAINS = {
    "crm": {
        "label": "CRM",
        "entities": {
            "contact": {
                "id_param": ("contact_id", "Unique contact identifier"),
                "specific_params": {
                    "email": {"type": "string", "description": "Contact email address"},
                    "phone": {"type": "string", "description": "Contact phone number"},
                    "company": {"type": "string", "description": "Company the contact belongs to"},
                    "lifecycle_stage": {"type": "string", "enum": ["lead", "mql", "sql", "opportunity", "customer", "churned"], "description": "Contact lifecycle stage"},
                    "owner_id": {"type": "string", "description": "ID of the sales rep who owns this contact"},
                },
            },
            "deal": {
                "id_param": ("deal_id", "Unique deal/opportunity identifier"),
                "specific_params": {
                    "pipeline_id": {"type": "string", "description": "Sales pipeline identifier"},
                    "stage": {"type": "string", "enum": ["prospecting", "qualification", "proposal", "negotiation", "closed_won", "closed_lost"], "description": "Current deal stage"},
                    "amount": {"type": "number", "description": "Deal value in the account currency"},
                    "close_date": {"type": "string", "description": "Expected close date (YYYY-MM-DD)"},
                    "contact_id": {"type": "string", "description": "Primary contact associated with this deal"},
                },
            },
            "account": {
                "id_param": ("account_id", "Unique account/company identifier"),
                "specific_params": {
                    "industry": {"type": "string", "description": "Industry vertical of the account"},
                    "employee_count": {"type": "integer", "description": "Number of employees"},
                    "annual_revenue": {"type": "number", "description": "Annual revenue in USD"},
                    "tier": {"type": "string", "enum": ["enterprise", "mid_market", "smb", "startup"], "description": "Account tier classification"},
                    "region": {"type": "string", "description": "Geographic region of the account"},
                },
            },
            "activity": {
                "id_param": ("activity_id", "Unique activity identifier"),
                "specific_params": {
                    "activity_type": {"type": "string", "enum": ["call", "email", "meeting", "note", "task"], "description": "Type of activity"},
                    "subject": {"type": "string", "description": "Activity subject line"},
                    "contact_id": {"type": "string", "description": "Contact this activity relates to"},
                    "due_date": {"type": "string", "description": "Due date for the activity (YYYY-MM-DD)"},
                    "completed": {"type": "boolean", "description": "Whether the activity has been completed"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve details for a specific {E} by ID. Returns all fields including custom properties and associated records."),
                ("search_{e}s", "Search for {E}s matching the specified filters. Supports full-text search across name and description fields."),
                ("list_{e}s", "List all {E}s with optional filtering and pagination. Results are sorted by last modified date by default."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} record with the specified properties. Returns the created {E} with its assigned ID."),
                ("update_{e}", "Update an existing {E}'s properties. Only specified fields are modified; unset fields remain unchanged."),
                ("assign_{e}", "Reassign a {E} to a different owner. Transfers all associated records and notifications."),
            ],
            "dangerous": [
                ("delete_{e}", "Permanently delete a {E} and all associated records. This action cannot be undone."),
                ("merge_{e}s", "Merge two duplicate {E} records into one. The secondary record is permanently deleted after merging."),
            ],
            "check": [
                ("check_{e}_duplicates", "Check for potential duplicate {E}s based on matching criteria. Returns a list of likely duplicates with confidence scores."),
            ],
        },
    },
    "billing": {
        "label": "Billing",
        "entities": {
            "invoice": {
                "id_param": ("invoice_id", "Unique invoice identifier"),
                "specific_params": {
                    "customer_id": {"type": "string", "description": "Customer this invoice belongs to"},
                    "status": {"type": "string", "enum": ["draft", "sent", "paid", "overdue", "void"], "description": "Current invoice status"},
                    "amount": {"type": "number", "description": "Total invoice amount"},
                    "currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "JPY"], "description": "Invoice currency code"},
                    "due_date": {"type": "string", "description": "Payment due date (YYYY-MM-DD)"},
                    "line_items": {"type": "array", "items": {"type": "object"}, "description": "Invoice line items with descriptions and amounts"},
                },
            },
            "subscription": {
                "id_param": ("subscription_id", "Unique subscription identifier"),
                "specific_params": {
                    "plan_id": {"type": "string", "description": "Billing plan identifier"},
                    "status": {"type": "string", "enum": ["active", "trialing", "past_due", "canceled", "paused"], "description": "Subscription status"},
                    "billing_cycle": {"type": "string", "enum": ["monthly", "quarterly", "annual"], "description": "Billing frequency"},
                    "customer_id": {"type": "string", "description": "Customer who owns this subscription"},
                    "next_billing_date": {"type": "string", "description": "Next scheduled billing date"},
                },
            },
            "payment": {
                "id_param": ("payment_id", "Unique payment identifier"),
                "specific_params": {
                    "amount": {"type": "number", "description": "Payment amount"},
                    "method": {"type": "string", "enum": ["credit_card", "bank_transfer", "ach", "wire", "check"], "description": "Payment method used"},
                    "invoice_id": {"type": "string", "description": "Invoice this payment applies to"},
                    "customer_id": {"type": "string", "description": "Customer who made the payment"},
                    "status": {"type": "string", "enum": ["pending", "completed", "failed", "refunded"], "description": "Payment processing status"},
                },
            },
            "refund": {
                "id_param": ("refund_id", "Unique refund identifier"),
                "specific_params": {
                    "payment_id": {"type": "string", "description": "Original payment to refund"},
                    "amount": {"type": "number", "description": "Refund amount (partial or full)"},
                    "reason": {"type": "string", "enum": ["duplicate", "product_not_received", "customer_request", "billing_error"], "description": "Reason for the refund"},
                    "customer_id": {"type": "string", "description": "Customer receiving the refund"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve a specific {E} by its ID. Returns full details including line items and payment history."),
                ("search_{e}s", "Search for {E}s by customer, date range, or status. Returns matching records with pagination."),
                ("list_{e}s", "List all {E}s with optional filters. Results include summary totals and status breakdowns."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} with the specified details. Validates all amounts and references before creation."),
                ("update_{e}", "Update an existing {E}'s fields. Cannot modify finalized or processed records."),
            ],
            "dangerous": [
                ("void_{e}", "Void a {E}, reversing all associated accounting entries. Cannot be undone once processed."),
                ("delete_{e}", "Permanently delete a {E} record. Only allowed for draft records that haven't been finalized."),
            ],
            "check": [
                ("validate_{e}", "Validate a {E}'s data integrity and business rules. Returns a list of errors or warnings if any."),
            ],
        },
    },
    "hr": {
        "label": "HR",
        "entities": {
            "employee": {
                "id_param": ("employee_id", "Unique employee identifier"),
                "specific_params": {
                    "department": {"type": "string", "description": "Department the employee belongs to"},
                    "title": {"type": "string", "description": "Job title"},
                    "manager_id": {"type": "string", "description": "Direct manager's employee ID"},
                    "employment_type": {"type": "string", "enum": ["full_time", "part_time", "contractor", "intern"], "description": "Employment type"},
                    "start_date": {"type": "string", "description": "Employment start date (YYYY-MM-DD)"},
                    "location": {"type": "string", "description": "Office location or remote status"},
                },
            },
            "leave_request": {
                "id_param": ("request_id", "Unique leave request identifier"),
                "specific_params": {
                    "employee_id": {"type": "string", "description": "Employee requesting leave"},
                    "leave_type": {"type": "string", "enum": ["vacation", "sick", "personal", "bereavement", "parental"], "description": "Type of leave"},
                    "start_date": {"type": "string", "description": "Leave start date"},
                    "end_date": {"type": "string", "description": "Leave end date"},
                    "status": {"type": "string", "enum": ["pending", "approved", "denied", "canceled"], "description": "Request approval status"},
                },
            },
            "performance_review": {
                "id_param": ("review_id", "Unique performance review identifier"),
                "specific_params": {
                    "employee_id": {"type": "string", "description": "Employee being reviewed"},
                    "reviewer_id": {"type": "string", "description": "Manager or peer conducting the review"},
                    "period": {"type": "string", "description": "Review period (e.g., Q1 2026, H1 2026)"},
                    "rating": {"type": "number", "description": "Overall performance rating (1.0-5.0)"},
                    "goals": {"type": "array", "items": {"type": "object"}, "description": "Goals and their completion status"},
                },
            },
            "job_posting": {
                "id_param": ("posting_id", "Unique job posting identifier"),
                "specific_params": {
                    "title": {"type": "string", "description": "Job title for the posting"},
                    "department": {"type": "string", "description": "Hiring department"},
                    "status": {"type": "string", "enum": ["draft", "open", "closed", "on_hold"], "description": "Posting status"},
                    "salary_range_min": {"type": "number", "description": "Minimum salary for the position"},
                    "salary_range_max": {"type": "number", "description": "Maximum salary for the position"},
                    "location": {"type": "string", "description": "Job location or remote"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve detailed information for a specific {E}. Includes all associated metadata and history."),
                ("search_{e}s", "Search {E}s by keyword, department, or status filters. Supports partial matching on text fields."),
                ("list_{e}s", "List all {E}s with optional filtering and sorting. Returns paginated results."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} record. Validates required fields and business rules before saving."),
                ("update_{e}", "Update fields on an existing {E}. Changes are logged in the audit trail."),
                ("approve_{e}", "Approve a pending {E}. Sends notification to the requesting employee."),
            ],
            "dangerous": [
                ("terminate_{e}", "Mark a {E} as terminated. Triggers offboarding workflows and access revocation."),
                ("delete_{e}", "Permanently remove a {E} record. Only available for draft or test records."),
            ],
            "check": [
                ("audit_{e}_history", "Retrieve the full change history for a {E}. Shows who changed what and when."),
            ],
        },
    },
    "it_devops": {
        "label": "IT/DevOps",
        "entities": {
            "incident": {
                "id_param": ("incident_id", "Unique incident identifier"),
                "specific_params": {
                    "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"], "description": "Incident severity level"},
                    "status": {"type": "string", "enum": ["open", "investigating", "mitigated", "resolved", "closed"], "description": "Current incident status"},
                    "service": {"type": "string", "description": "Affected service or system"},
                    "assignee_id": {"type": "string", "description": "On-call engineer assigned to this incident"},
                    "description": {"type": "string", "description": "Detailed description of the incident"},
                },
            },
            "deployment": {
                "id_param": ("deployment_id", "Unique deployment identifier"),
                "specific_params": {
                    "service": {"type": "string", "description": "Service being deployed"},
                    "version": {"type": "string", "description": "Version or commit SHA being deployed"},
                    "environment": {"type": "string", "enum": ["development", "staging", "production"], "description": "Target deployment environment"},
                    "strategy": {"type": "string", "enum": ["rolling", "blue_green", "canary"], "description": "Deployment strategy"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "failed", "rolled_back"], "description": "Deployment status"},
                },
            },
            "server": {
                "id_param": ("server_id", "Unique server or instance identifier"),
                "specific_params": {
                    "hostname": {"type": "string", "description": "Server hostname"},
                    "ip_address": {"type": "string", "description": "Server IP address"},
                    "instance_type": {"type": "string", "description": "Cloud instance type (e.g., m5.xlarge)"},
                    "region": {"type": "string", "description": "Cloud region or data center location"},
                    "status": {"type": "string", "enum": ["running", "stopped", "terminated", "maintenance"], "description": "Server status"},
                    "cpu_utilization": {"type": "number", "description": "Current CPU utilization percentage"},
                },
            },
            "change_request": {
                "id_param": ("change_id", "Unique change request identifier"),
                "specific_params": {
                    "title": {"type": "string", "description": "Short description of the change"},
                    "change_type": {"type": "string", "enum": ["standard", "normal", "emergency"], "description": "Type of change request"},
                    "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"], "description": "Assessed risk level"},
                    "scheduled_start": {"type": "string", "description": "Scheduled start time (ISO 8601)"},
                    "approver_id": {"type": "string", "description": "Change advisory board approver"},
                    "rollback_plan": {"type": "string", "description": "Description of the rollback procedure"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve full details for a specific {E} including timeline and associated records."),
                ("search_{e}s", "Search {E}s by service, status, severity, or date range. Returns matching records with context."),
                ("list_{e}s", "List all {E}s with optional status and date filters. Results include summary metrics."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} record. Triggers relevant notification workflows based on severity."),
                ("update_{e}", "Update an existing {E}'s fields. All changes are logged with timestamps."),
            ],
            "dangerous": [
                ("rollback_{e}", "Roll back a {E} to its previous state. May cause service disruption during the rollback window."),
                ("force_close_{e}", "Force-close a {E} without completing standard resolution steps. Requires manager approval."),
            ],
            "check": [
                ("check_{e}_status", "Get the current status and health metrics for a {E}. Includes uptime and error rate data."),
            ],
        },
    },
    "compliance": {
        "label": "Compliance",
        "entities": {
            "policy": {
                "id_param": ("policy_id", "Unique compliance policy identifier"),
                "specific_params": {
                    "title": {"type": "string", "description": "Policy title"},
                    "framework": {"type": "string", "enum": ["SOC2", "GDPR", "HIPAA", "PCI_DSS", "ISO27001", "SOX"], "description": "Compliance framework this policy belongs to"},
                    "status": {"type": "string", "enum": ["draft", "active", "under_review", "deprecated"], "description": "Policy status"},
                    "owner_id": {"type": "string", "description": "Policy owner (compliance officer)"},
                    "review_date": {"type": "string", "description": "Next scheduled review date"},
                },
            },
            "audit_record": {
                "id_param": ("audit_id", "Unique audit identifier"),
                "specific_params": {
                    "audit_type": {"type": "string", "enum": ["internal", "external", "regulatory"], "description": "Type of audit"},
                    "scope": {"type": "string", "description": "Systems and processes in scope for the audit"},
                    "status": {"type": "string", "enum": ["planned", "in_progress", "findings_review", "completed"], "description": "Audit status"},
                    "lead_auditor": {"type": "string", "description": "Lead auditor name or ID"},
                    "findings_count": {"type": "integer", "description": "Number of findings identified"},
                },
            },
            "violation": {
                "id_param": ("violation_id", "Unique violation identifier"),
                "specific_params": {
                    "policy_id": {"type": "string", "description": "Policy that was violated"},
                    "severity": {"type": "string", "enum": ["critical", "major", "minor", "observation"], "description": "Violation severity"},
                    "description": {"type": "string", "description": "Detailed description of the violation"},
                    "remediation_deadline": {"type": "string", "description": "Deadline for remediation (YYYY-MM-DD)"},
                    "assigned_to": {"type": "string", "description": "Person responsible for remediation"},
                },
            },
            "consent_record": {
                "id_param": ("consent_id", "Unique consent record identifier"),
                "specific_params": {
                    "subject_id": {"type": "string", "description": "Data subject (customer/user) identifier"},
                    "purpose": {"type": "string", "enum": ["marketing", "analytics", "essential", "third_party_sharing"], "description": "Purpose of data processing consent"},
                    "granted": {"type": "boolean", "description": "Whether consent was granted"},
                    "channel": {"type": "string", "enum": ["web", "email", "phone", "in_person"], "description": "Channel through which consent was collected"},
                    "expiry_date": {"type": "string", "description": "Date consent expires (if applicable)"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve full details of a specific {E} including version history and related records."),
                ("search_{e}s", "Search {E}s by framework, status, or keyword. Returns matching records with compliance context."),
                ("list_{e}s", "List all {E}s with filtering options. Includes summary statistics and status breakdown."),
            ],
            "write": [
                ("create_{e}", "Create a new {E}. Validates against the applicable compliance framework requirements."),
                ("update_{e}", "Update a {E}'s fields. Creates a new version entry in the audit log."),
            ],
            "dangerous": [
                ("revoke_{e}", "Revoke or invalidate a {E}. This may trigger downstream compliance notifications and remediation workflows."),
                ("purge_{e}", "Permanently delete a {E} and all associated data. Required for GDPR right-to-erasure requests. Cannot be undone."),
            ],
            "check": [
                ("verify_{e}_compliance", "Check whether a {E} meets current compliance requirements. Returns pass/fail with detailed findings."),
            ],
        },
    },
    "analytics": {
        "label": "Analytics",
        "entities": {
            "report": {
                "id_param": ("report_id", "Unique report identifier"),
                "specific_params": {
                    "report_type": {"type": "string", "enum": ["dashboard", "scheduled", "ad_hoc", "regulatory"], "description": "Type of report"},
                    "data_source": {"type": "string", "description": "Primary data source for the report"},
                    "format": {"type": "string", "enum": ["pdf", "csv", "excel", "json"], "description": "Output format"},
                    "schedule": {"type": "string", "description": "Cron expression for scheduled reports"},
                    "recipients": {"type": "array", "items": {"type": "string"}, "description": "Email addresses to receive the report"},
                },
            },
            "metric": {
                "id_param": ("metric_id", "Unique metric identifier"),
                "specific_params": {
                    "metric_name": {"type": "string", "description": "Name of the business metric"},
                    "aggregation": {"type": "string", "enum": ["sum", "avg", "count", "min", "max", "p95", "p99"], "description": "Aggregation function"},
                    "granularity": {"type": "string", "enum": ["minute", "hour", "day", "week", "month"], "description": "Time granularity for data points"},
                    "dimension": {"type": "string", "description": "Dimension to group results by (e.g., region, product)"},
                    "threshold": {"type": "number", "description": "Alert threshold value"},
                },
            },
            "query": {
                "id_param": ("query_id", "Unique saved query identifier"),
                "specific_params": {
                    "sql": {"type": "string", "description": "SQL query string"},
                    "database": {"type": "string", "description": "Target database or warehouse"},
                    "timeout_seconds": {"type": "integer", "description": "Query execution timeout in seconds"},
                    "cache_ttl": {"type": "integer", "description": "Cache time-to-live in seconds"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve a saved {E} by ID, including its configuration and last execution metadata."),
                ("search_{e}s", "Search for {E}s by name, type, or data source. Returns matching {E}s with usage statistics."),
                ("list_{e}s", "List all available {E}s. Includes last-run timestamps and owner information."),
                ("run_{e}", "Execute a {E} and return the results. For large result sets, returns a download URL."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} with the specified configuration. Validates syntax and permissions before saving."),
                ("update_{e}", "Update an existing {E}'s definition or schedule. Previous versions are retained."),
            ],
            "dangerous": [
                ("delete_{e}", "Permanently delete a saved {E} and all cached results. Scheduled deliveries will stop immediately."),
            ],
            "check": [
                ("validate_{e}", "Validate a {E}'s syntax and data source connectivity without executing it. Returns errors if any."),
            ],
        },
    },
    "communication": {
        "label": "Communication",
        "entities": {
            "message": {
                "id_param": ("message_id", "Unique message identifier"),
                "specific_params": {
                    "channel_id": {"type": "string", "description": "Channel or conversation to post to"},
                    "content": {"type": "string", "description": "Message body text (supports markdown)"},
                    "sender_id": {"type": "string", "description": "User sending the message"},
                    "priority": {"type": "string", "enum": ["normal", "high", "urgent"], "description": "Message priority level"},
                    "thread_id": {"type": "string", "description": "Thread ID to reply to (for threaded conversations)"},
                    "attachments": {"type": "array", "items": {"type": "object"}, "description": "File attachments"},
                },
            },
            "channel": {
                "id_param": ("channel_id", "Unique channel identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Channel display name"},
                    "visibility": {"type": "string", "enum": ["public", "private", "shared"], "description": "Channel visibility setting"},
                    "team_id": {"type": "string", "description": "Team that owns this channel"},
                    "topic": {"type": "string", "description": "Channel topic or description"},
                    "member_count": {"type": "integer", "description": "Number of members in the channel"},
                },
            },
            "notification": {
                "id_param": ("notification_id", "Unique notification identifier"),
                "specific_params": {
                    "recipient_id": {"type": "string", "description": "User to notify"},
                    "template_id": {"type": "string", "description": "Notification template identifier"},
                    "delivery_channel": {"type": "string", "enum": ["email", "sms", "push", "in_app", "slack"], "description": "Delivery channel"},
                    "subject": {"type": "string", "description": "Notification subject line"},
                    "body": {"type": "string", "description": "Notification body content"},
                    "scheduled_at": {"type": "string", "description": "When to send (ISO 8601), or null for immediate"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve a specific {E} by ID, including delivery status and read receipts."),
                ("search_{e}s", "Search {E}s by sender, content keyword, or date range. Returns matching results with context."),
                ("list_{e}s", "List {E}s with optional filters for channel, sender, and date. Supports cursor-based pagination."),
            ],
            "write": [
                ("send_{e}", "Send a new {E} to the specified recipients or channel. Returns the created {E} with delivery tracking."),
                ("update_{e}", "Edit an existing {E}'s content or metadata. Marks the {E} as edited."),
                ("schedule_{e}", "Schedule a {E} for future delivery. Can be canceled before the scheduled time."),
            ],
            "dangerous": [
                ("delete_{e}", "Permanently delete a {E}. Recipients who already received it will see a deletion notice."),
                ("purge_{e}s", "Bulk delete all {E}s matching the specified criteria. This action cannot be undone."),
            ],
            "check": [
                ("check_{e}_delivery", "Check the delivery status of a sent {E}. Returns per-recipient delivery and read status."),
            ],
        },
    },
    "project_management": {
        "label": "Project Management",
        "entities": {
            "task": {
                "id_param": ("task_id", "Unique task identifier"),
                "specific_params": {
                    "title": {"type": "string", "description": "Task title"},
                    "project_id": {"type": "string", "description": "Project this task belongs to"},
                    "assignee_id": {"type": "string", "description": "User assigned to this task"},
                    "status": {"type": "string", "enum": ["backlog", "todo", "in_progress", "in_review", "done"], "description": "Task status"},
                    "priority": {"type": "string", "enum": ["critical", "high", "medium", "low"], "description": "Task priority"},
                    "due_date": {"type": "string", "description": "Task due date (YYYY-MM-DD)"},
                    "story_points": {"type": "integer", "description": "Story point estimate"},
                },
            },
            "project": {
                "id_param": ("project_id", "Unique project identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Project name"},
                    "owner_id": {"type": "string", "description": "Project owner/manager"},
                    "status": {"type": "string", "enum": ["planning", "active", "on_hold", "completed", "archived"], "description": "Project status"},
                    "budget": {"type": "number", "description": "Project budget in USD"},
                    "deadline": {"type": "string", "description": "Project deadline (YYYY-MM-DD)"},
                    "team_ids": {"type": "array", "items": {"type": "string"}, "description": "Team member IDs"},
                },
            },
            "sprint": {
                "id_param": ("sprint_id", "Unique sprint identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Sprint name (e.g., Sprint 24)"},
                    "project_id": {"type": "string", "description": "Project this sprint belongs to"},
                    "start_date": {"type": "string", "description": "Sprint start date"},
                    "end_date": {"type": "string", "description": "Sprint end date"},
                    "goal": {"type": "string", "description": "Sprint goal statement"},
                    "velocity": {"type": "integer", "description": "Planned velocity in story points"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve detailed information about a specific {E} including subtasks and activity history."),
                ("search_{e}s", "Search for {E}s by title, assignee, status, or label. Returns matching results with project context."),
                ("list_{e}s", "List all {E}s with optional filtering by project, sprint, or status. Supports pagination."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} with the provided details. Sends notifications to relevant team members."),
                ("update_{e}", "Update an existing {E}'s fields. Logs the change in the {E}'s activity timeline."),
                ("assign_{e}", "Assign or reassign a {E} to a team member. Sends a notification to the new assignee."),
            ],
            "dangerous": [
                ("delete_{e}", "Permanently delete a {E} and all associated subtasks. This action cannot be undone."),
                ("archive_{e}", "Archive a {E}, removing it from active views. Archived {E}s can be restored within 90 days."),
            ],
            "check": [
                ("check_{e}_progress", "Get progress metrics for a {E} including completion percentage, velocity, and burndown data."),
            ],
        },
    },
    "security": {
        "label": "Security",
        "entities": {
            "threat": {
                "id_param": ("threat_id", "Unique threat identifier"),
                "specific_params": {
                    "threat_type": {"type": "string", "enum": ["malware", "phishing", "brute_force", "insider_threat", "data_exfiltration", "ransomware"], "description": "Type of threat detected"},
                    "severity": {"type": "string", "enum": ["critical", "high", "medium", "low", "informational"], "description": "Threat severity rating"},
                    "source_ip": {"type": "string", "description": "Source IP address of the threat"},
                    "affected_systems": {"type": "array", "items": {"type": "string"}, "description": "Systems affected by the threat"},
                    "status": {"type": "string", "enum": ["detected", "investigating", "contained", "remediated", "false_positive"], "description": "Current threat status"},
                },
            },
            "vulnerability": {
                "id_param": ("vulnerability_id", "Unique vulnerability identifier (CVE or internal)"),
                "specific_params": {
                    "cve_id": {"type": "string", "description": "CVE identifier if applicable"},
                    "cvss_score": {"type": "number", "description": "CVSS severity score (0.0-10.0)"},
                    "affected_component": {"type": "string", "description": "Software component or service affected"},
                    "status": {"type": "string", "enum": ["open", "in_progress", "patched", "accepted_risk", "false_positive"], "description": "Remediation status"},
                    "patch_available": {"type": "boolean", "description": "Whether a patch is available"},
                },
            },
            "security_event": {
                "id_param": ("event_id", "Unique security event identifier"),
                "specific_params": {
                    "event_type": {"type": "string", "enum": ["login_failure", "privilege_escalation", "file_access", "config_change", "network_anomaly"], "description": "Type of security event"},
                    "user_id": {"type": "string", "description": "User involved in the event"},
                    "source_ip": {"type": "string", "description": "IP address associated with the event"},
                    "resource": {"type": "string", "description": "Resource or system that was accessed"},
                    "risk_score": {"type": "number", "description": "Calculated risk score (0-100)"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve full details of a {E} including indicators of compromise and timeline data."),
                ("search_{e}s", "Search {E}s by type, severity, IP, or date range. Returns correlated results with risk context."),
                ("list_{e}s", "List all {E}s with optional severity and status filters. Includes risk score summaries."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} record. Automatically triggers severity-based alerting and escalation rules."),
                ("update_{e}", "Update a {E}'s status or details. All changes are cryptographically logged."),
            ],
            "dangerous": [
                ("quarantine_{e}", "Quarantine the source of a {E} by isolating affected systems from the network. May cause service disruption."),
                ("block_{e}_source", "Block the source IP or user associated with a {E}. Immediately revokes all active sessions."),
            ],
            "check": [
                ("assess_{e}_risk", "Run a risk assessment on a {E}. Returns risk score, blast radius estimate, and recommended actions."),
            ],
        },
    },
    "erp_supply_chain": {
        "label": "ERP/Supply Chain",
        "entities": {
            "purchase_order": {
                "id_param": ("po_id", "Unique purchase order identifier"),
                "specific_params": {
                    "vendor_id": {"type": "string", "description": "Vendor/supplier identifier"},
                    "status": {"type": "string", "enum": ["draft", "submitted", "approved", "shipped", "received", "closed"], "description": "Purchase order status"},
                    "total_amount": {"type": "number", "description": "Total order amount"},
                    "currency": {"type": "string", "description": "Order currency code"},
                    "delivery_date": {"type": "string", "description": "Expected delivery date"},
                    "line_items": {"type": "array", "items": {"type": "object"}, "description": "Order line items with quantities and prices"},
                },
            },
            "inventory_item": {
                "id_param": ("item_id", "Unique inventory item identifier (SKU)"),
                "specific_params": {
                    "sku": {"type": "string", "description": "Stock keeping unit code"},
                    "warehouse_id": {"type": "string", "description": "Warehouse location identifier"},
                    "quantity_on_hand": {"type": "integer", "description": "Current quantity in stock"},
                    "reorder_point": {"type": "integer", "description": "Quantity threshold to trigger reorder"},
                    "unit_cost": {"type": "number", "description": "Unit cost in base currency"},
                    "category": {"type": "string", "description": "Product category"},
                },
            },
            "shipment": {
                "id_param": ("shipment_id", "Unique shipment identifier"),
                "specific_params": {
                    "carrier": {"type": "string", "description": "Shipping carrier name"},
                    "tracking_number": {"type": "string", "description": "Carrier tracking number"},
                    "origin": {"type": "string", "description": "Origin warehouse or facility"},
                    "destination": {"type": "string", "description": "Destination address or facility"},
                    "status": {"type": "string", "enum": ["pending", "in_transit", "delivered", "returned", "lost"], "description": "Shipment status"},
                    "estimated_arrival": {"type": "string", "description": "Estimated arrival date and time (ISO 8601)"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve full details of a {E} including line items, status history, and associated documents."),
                ("search_{e}s", "Search {E}s by vendor, status, date, or amount range. Returns matching records with logistics context."),
                ("list_{e}s", "List all {E}s with optional status and date filters. Includes summary totals and aging data."),
                ("track_{e}", "Get real-time tracking information for a {E} including current location and estimated delivery."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} with the specified details. Validates vendor and item references."),
                ("update_{e}", "Update an existing {E}'s details. Cannot modify closed or finalized records."),
            ],
            "dangerous": [
                ("cancel_{e}", "Cancel a {E}. Triggers reversal of accounting entries and notifies all parties."),
                ("write_off_{e}", "Write off a {E} as a loss. Creates the corresponding accounting entries and audit trail."),
            ],
            "check": [
                ("verify_{e}_receipt", "Verify receipt of a {E} against the purchase order. Reports discrepancies in quantity or condition."),
            ],
        },
    },
    "healthcare": {
        "label": "Healthcare",
        "entities": {
            "patient": {
                "id_param": ("patient_id", "Unique patient identifier (MRN)"),
                "specific_params": {
                    "mrn": {"type": "string", "description": "Medical record number"},
                    "date_of_birth": {"type": "string", "description": "Patient date of birth (YYYY-MM-DD)"},
                    "insurance_id": {"type": "string", "description": "Insurance plan identifier"},
                    "primary_provider_id": {"type": "string", "description": "Primary care provider NPI"},
                    "allergies": {"type": "array", "items": {"type": "string"}, "description": "Known allergies"},
                    "status": {"type": "string", "enum": ["active", "inactive", "deceased"], "description": "Patient record status"},
                },
            },
            "appointment": {
                "id_param": ("appointment_id", "Unique appointment identifier"),
                "specific_params": {
                    "patient_id": {"type": "string", "description": "Patient being seen"},
                    "provider_id": {"type": "string", "description": "Healthcare provider NPI"},
                    "appointment_type": {"type": "string", "enum": ["initial", "follow_up", "urgent", "telehealth", "procedure"], "description": "Type of appointment"},
                    "scheduled_time": {"type": "string", "description": "Appointment date and time (ISO 8601)"},
                    "duration_minutes": {"type": "integer", "description": "Expected duration in minutes"},
                    "status": {"type": "string", "enum": ["scheduled", "checked_in", "in_progress", "completed", "no_show", "canceled"], "description": "Appointment status"},
                },
            },
            "prescription": {
                "id_param": ("prescription_id", "Unique prescription identifier"),
                "specific_params": {
                    "patient_id": {"type": "string", "description": "Patient the prescription is for"},
                    "medication": {"type": "string", "description": "Medication name and strength"},
                    "dosage": {"type": "string", "description": "Dosage instructions"},
                    "prescriber_id": {"type": "string", "description": "Prescribing provider NPI"},
                    "refills_remaining": {"type": "integer", "description": "Number of refills remaining"},
                    "pharmacy_id": {"type": "string", "description": "Dispensing pharmacy identifier"},
                },
            },
            "lab_order": {
                "id_param": ("order_id", "Unique lab order identifier"),
                "specific_params": {
                    "patient_id": {"type": "string", "description": "Patient the lab work is for"},
                    "test_codes": {"type": "array", "items": {"type": "string"}, "description": "Lab test codes (e.g., CBC, BMP, TSH)"},
                    "ordering_provider_id": {"type": "string", "description": "Provider who ordered the tests"},
                    "priority": {"type": "string", "enum": ["routine", "stat", "urgent"], "description": "Order priority"},
                    "status": {"type": "string", "enum": ["ordered", "collected", "processing", "resulted", "canceled"], "description": "Lab order status"},
                    "fasting_required": {"type": "boolean", "description": "Whether fasting is required before collection"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve complete {E} information including history and associated records. Requires HIPAA-compliant access."),
                ("search_{e}s", "Search {E}s by patient, provider, date, or status. All queries are logged for HIPAA compliance."),
                ("list_{e}s", "List {E}s with optional filters. Returns paginated results with PHI access logging."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} record. Validates clinical data and insurance eligibility before saving."),
                ("update_{e}", "Update a {E}'s details. Creates a timestamped amendment record per HIPAA requirements."),
            ],
            "dangerous": [
                ("cancel_{e}", "Cancel a {E}. Sends notifications to the patient and all associated providers."),
                ("purge_{e}_data", "Permanently delete a {E}'s protected health information. Requires dual authorization and is logged for compliance."),
            ],
            "check": [
                ("verify_{e}_eligibility", "Verify insurance eligibility and coverage for a {E}. Returns coverage details and copay estimates."),
            ],
        },
    },
    "finance": {
        "label": "Finance",
        "entities": {
            "transaction": {
                "id_param": ("transaction_id", "Unique transaction identifier"),
                "specific_params": {
                    "account_id": {"type": "string", "description": "Financial account identifier"},
                    "amount": {"type": "number", "description": "Transaction amount (positive=credit, negative=debit)"},
                    "currency": {"type": "string", "description": "Transaction currency (ISO 4217)"},
                    "transaction_type": {"type": "string", "enum": ["deposit", "withdrawal", "transfer", "fee", "interest", "adjustment"], "description": "Transaction type"},
                    "counterparty": {"type": "string", "description": "Other party in the transaction"},
                    "reference": {"type": "string", "description": "External reference number"},
                },
            },
            "budget": {
                "id_param": ("budget_id", "Unique budget identifier"),
                "specific_params": {
                    "department_id": {"type": "string", "description": "Department this budget is for"},
                    "fiscal_year": {"type": "string", "description": "Fiscal year (e.g., FY2026)"},
                    "total_amount": {"type": "number", "description": "Total budgeted amount"},
                    "spent_amount": {"type": "number", "description": "Amount spent to date"},
                    "category": {"type": "string", "enum": ["opex", "capex", "personnel", "marketing", "r_and_d"], "description": "Budget category"},
                    "status": {"type": "string", "enum": ["draft", "approved", "active", "frozen", "closed"], "description": "Budget status"},
                },
            },
            "expense_report": {
                "id_param": ("expense_id", "Unique expense report identifier"),
                "specific_params": {
                    "submitter_id": {"type": "string", "description": "Employee who submitted the expense"},
                    "total_amount": {"type": "number", "description": "Total expense amount"},
                    "status": {"type": "string", "enum": ["draft", "submitted", "approved", "rejected", "reimbursed"], "description": "Expense report status"},
                    "category": {"type": "string", "enum": ["travel", "meals", "supplies", "software", "conference", "other"], "description": "Expense category"},
                    "receipts": {"type": "array", "items": {"type": "object"}, "description": "Attached receipt images or documents"},
                    "approver_id": {"type": "string", "description": "Manager who approved the expense"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve a {E} by ID with full details, including audit trail and approval chain."),
                ("search_{e}s", "Search {E}s by account, amount range, date, or type. Returns matching records with running balances."),
                ("list_{e}s", "List {E}s with optional filters for date, status, and category. Includes period totals."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} record. Validates against budget limits and approval policies."),
                ("update_{e}", "Modify an existing {E}. Only allowed for draft or pending records."),
                ("approve_{e}", "Approve a pending {E}. Routes to the next approver if multi-level approval is required."),
            ],
            "dangerous": [
                ("reverse_{e}", "Reverse a finalized {E}. Creates an offsetting entry and requires supervisor authorization."),
                ("delete_{e}", "Permanently delete a {E} record. Only allowed for draft records that haven't entered the approval flow."),
            ],
            "check": [
                ("audit_{e}_trail", "Run an audit check on a {E}. Validates amounts, coding, and policy compliance. Returns discrepancies."),
            ],
        },
    },
    "document_management": {
        "label": "Document Management",
        "entities": {
            "document": {
                "id_param": ("document_id", "Unique document identifier"),
                "specific_params": {
                    "title": {"type": "string", "description": "Document title"},
                    "file_type": {"type": "string", "enum": ["pdf", "docx", "xlsx", "pptx", "csv", "txt"], "description": "Document file format"},
                    "folder_id": {"type": "string", "description": "Parent folder identifier"},
                    "owner_id": {"type": "string", "description": "Document owner user ID"},
                    "version": {"type": "integer", "description": "Current version number"},
                    "size_bytes": {"type": "integer", "description": "File size in bytes"},
                    "classification": {"type": "string", "enum": ["public", "internal", "confidential", "restricted"], "description": "Data classification level"},
                },
            },
            "folder": {
                "id_param": ("folder_id", "Unique folder identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Folder name"},
                    "parent_id": {"type": "string", "description": "Parent folder ID (null for root)"},
                    "owner_id": {"type": "string", "description": "Folder owner user ID"},
                    "shared_with": {"type": "array", "items": {"type": "string"}, "description": "User or group IDs with access"},
                },
            },
            "template": {
                "id_param": ("template_id", "Unique template identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Template name"},
                    "category": {"type": "string", "enum": ["contract", "proposal", "report", "memo", "form", "policy"], "description": "Template category"},
                    "placeholders": {"type": "array", "items": {"type": "string"}, "description": "List of placeholder variables in the template"},
                    "file_type": {"type": "string", "description": "Output file format"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve a {E} by ID including metadata, permissions, and version history."),
                ("search_{e}s", "Full-text search across {E}s by content, title, or metadata. Returns relevance-ranked results."),
                ("list_{e}s", "List {E}s in a specified location with optional type and date filters. Returns metadata only."),
                ("download_{e}", "Download the latest version of a {E}. Returns a time-limited signed URL."),
            ],
            "write": [
                ("upload_{e}", "Upload a new {E} to the specified folder. Automatically extracts metadata and indexes content."),
                ("update_{e}", "Update a {E}'s metadata or upload a new version. Previous versions are preserved."),
                ("share_{e}", "Share a {E} with specified users or groups with defined permission levels."),
            ],
            "dangerous": [
                ("delete_{e}", "Permanently delete a {E} and all its versions. Moves to trash with 30-day recovery window."),
                ("purge_{e}_versions", "Permanently delete all but the latest version of a {E}. Frees storage but removes version history."),
            ],
            "check": [
                ("check_{e}_permissions", "Check who has access to a {E} and at what permission level. Returns the full access list."),
            ],
        },
    },
    "customer_support": {
        "label": "Customer Support",
        "entities": {
            "ticket": {
                "id_param": ("ticket_id", "Unique support ticket identifier"),
                "specific_params": {
                    "subject": {"type": "string", "description": "Ticket subject line"},
                    "description": {"type": "string", "description": "Detailed issue description"},
                    "customer_id": {"type": "string", "description": "Customer who raised the ticket"},
                    "assignee_id": {"type": "string", "description": "Support agent assigned to the ticket"},
                    "priority": {"type": "string", "enum": ["urgent", "high", "normal", "low"], "description": "Ticket priority"},
                    "status": {"type": "string", "enum": ["new", "open", "pending", "on_hold", "solved", "closed"], "description": "Ticket status"},
                    "channel": {"type": "string", "enum": ["email", "chat", "phone", "web", "social"], "description": "Channel through which the ticket was created"},
                    "category": {"type": "string", "description": "Issue category (e.g., billing, technical, account)"},
                },
            },
            "knowledge_article": {
                "id_param": ("article_id", "Unique knowledge base article identifier"),
                "specific_params": {
                    "title": {"type": "string", "description": "Article title"},
                    "body": {"type": "string", "description": "Article content in markdown"},
                    "category": {"type": "string", "description": "Article category"},
                    "status": {"type": "string", "enum": ["draft", "published", "archived"], "description": "Publication status"},
                    "locale": {"type": "string", "description": "Article language/locale (e.g., en-US)"},
                    "helpful_votes": {"type": "integer", "description": "Number of helpful votes from users"},
                },
            },
            "sla": {
                "id_param": ("sla_id", "Unique SLA policy identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "SLA policy name"},
                    "priority": {"type": "string", "enum": ["urgent", "high", "normal", "low"], "description": "Priority level this SLA applies to"},
                    "first_response_hours": {"type": "integer", "description": "Maximum hours for first response"},
                    "resolution_hours": {"type": "integer", "description": "Maximum hours for resolution"},
                    "business_hours_only": {"type": "boolean", "description": "Whether SLA counts business hours only"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve a {E} by ID with full conversation history, SLA status, and linked records."),
                ("search_{e}s", "Search {E}s by keyword, customer, agent, or status. Returns matching results with SLA compliance data."),
                ("list_{e}s", "List {E}s with optional filters for priority, status, and assignee. Includes queue statistics."),
            ],
            "write": [
                ("create_{e}", "Create a new {E}. Automatically applies SLA policies and routing rules based on priority and category."),
                ("update_{e}", "Update a {E}'s fields. Triggers SLA recalculation if priority or assignee changes."),
                ("escalate_{e}", "Escalate a {E} to a higher support tier or manager. Adds escalation context and resets SLA timers."),
            ],
            "dangerous": [
                ("delete_{e}", "Permanently delete a {E} and all associated conversation data. Cannot be recovered."),
                ("bulk_close_{e}s", "Close all {E}s matching the specified criteria. Sends a resolution notification to each customer."),
            ],
            "check": [
                ("check_{e}_sla", "Check SLA compliance status for a {E}. Returns time remaining, breach risk, and recommended actions."),
            ],
        },
    },
    "marketing": {
        "label": "Marketing",
        "entities": {
            "campaign": {
                "id_param": ("campaign_id", "Unique marketing campaign identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Campaign name"},
                    "channel": {"type": "string", "enum": ["email", "social", "ppc", "display", "seo", "content"], "description": "Marketing channel"},
                    "status": {"type": "string", "enum": ["draft", "scheduled", "active", "paused", "completed"], "description": "Campaign status"},
                    "budget": {"type": "number", "description": "Campaign budget in USD"},
                    "start_date": {"type": "string", "description": "Campaign start date"},
                    "end_date": {"type": "string", "description": "Campaign end date"},
                    "target_audience": {"type": "string", "description": "Target audience segment description"},
                },
            },
            "lead": {
                "id_param": ("lead_id", "Unique lead identifier"),
                "specific_params": {
                    "email": {"type": "string", "description": "Lead email address"},
                    "source": {"type": "string", "enum": ["organic", "paid", "referral", "event", "outbound", "partner"], "description": "Lead acquisition source"},
                    "score": {"type": "integer", "description": "Lead score (0-100)"},
                    "status": {"type": "string", "enum": ["new", "contacted", "qualified", "unqualified", "converted"], "description": "Lead status"},
                    "campaign_id": {"type": "string", "description": "Campaign that generated this lead"},
                    "company": {"type": "string", "description": "Lead's company name"},
                },
            },
            "email_template": {
                "id_param": ("template_id", "Unique email template identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Template name"},
                    "subject_line": {"type": "string", "description": "Email subject line (supports merge fields)"},
                    "html_body": {"type": "string", "description": "HTML email body content"},
                    "category": {"type": "string", "enum": ["promotional", "transactional", "nurture", "newsletter", "announcement"], "description": "Template category"},
                    "merge_fields": {"type": "array", "items": {"type": "string"}, "description": "Available merge field variables"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve a {E} by ID with performance metrics, audience data, and content details."),
                ("search_{e}s", "Search {E}s by name, channel, status, or date range. Returns results with performance summaries."),
                ("list_{e}s", "List all {E}s with optional channel and status filters. Includes aggregate performance metrics."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} with the specified configuration. Validates audience targeting and budget limits."),
                ("update_{e}", "Update an existing {E}'s settings or content. Active {E}s may require re-approval after changes."),
                ("launch_{e}", "Launch or activate a {E}. Begins delivery to the target audience at the scheduled time."),
            ],
            "dangerous": [
                ("delete_{e}", "Permanently delete a {E} and all associated analytics data. This cannot be undone."),
                ("pause_{e}", "Immediately pause an active {E}. Stops all delivery and ad spend until resumed."),
            ],
            "check": [
                ("analyze_{e}_performance", "Run a performance analysis on a {E}. Returns ROI, conversion rates, and comparison to benchmarks."),
            ],
        },
    },
    "identity_access": {
        "label": "Identity & Access Management",
        "entities": {
            "user_account": {
                "id_param": ("user_id", "Unique user account identifier"),
                "specific_params": {
                    "username": {"type": "string", "description": "Username for login"},
                    "email": {"type": "string", "description": "User email address"},
                    "status": {"type": "string", "enum": ["active", "suspended", "locked", "deactivated"], "description": "Account status"},
                    "mfa_enabled": {"type": "boolean", "description": "Whether multi-factor authentication is enabled"},
                    "last_login": {"type": "string", "description": "Timestamp of last successful login"},
                    "department": {"type": "string", "description": "User's department"},
                },
            },
            "role": {
                "id_param": ("role_id", "Unique role identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Role name (e.g., admin, editor, viewer)"},
                    "permissions": {"type": "array", "items": {"type": "string"}, "description": "List of permission strings"},
                    "scope": {"type": "string", "enum": ["global", "organization", "project", "resource"], "description": "Permission scope level"},
                    "user_count": {"type": "integer", "description": "Number of users assigned this role"},
                    "description": {"type": "string", "description": "Human-readable role description"},
                },
            },
            "api_key": {
                "id_param": ("key_id", "Unique API key identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Descriptive name for the API key"},
                    "scopes": {"type": "array", "items": {"type": "string"}, "description": "API scopes/permissions granted"},
                    "expires_at": {"type": "string", "description": "Expiration date (ISO 8601), null for non-expiring"},
                    "created_by": {"type": "string", "description": "User who created the key"},
                    "last_used": {"type": "string", "description": "Timestamp of last use"},
                    "rate_limit": {"type": "integer", "description": "Requests per minute rate limit"},
                },
            },
            "access_policy": {
                "id_param": ("policy_id", "Unique access policy identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Policy name"},
                    "effect": {"type": "string", "enum": ["allow", "deny"], "description": "Whether the policy allows or denies access"},
                    "resources": {"type": "array", "items": {"type": "string"}, "description": "Resource patterns this policy applies to"},
                    "conditions": {"type": "object", "description": "Conditions that must be met for the policy to apply"},
                    "principals": {"type": "array", "items": {"type": "string"}, "description": "Users, roles, or groups the policy applies to"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve a {E} by ID including all associated permissions and audit history."),
                ("search_{e}s", "Search {E}s by name, status, or permission. Returns matching records with assignment details."),
                ("list_{e}s", "List all {E}s with optional scope and status filters. Includes usage statistics."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} with the specified configuration. Validates against organizational security policies."),
                ("update_{e}", "Update a {E}'s properties. Permission changes take effect immediately across all sessions."),
                ("assign_{e}", "Assign a {E} to users or groups. Propagates permissions to all associated resources."),
            ],
            "dangerous": [
                ("revoke_{e}", "Revoke a {E}, immediately removing all associated access. Active sessions using this {E} are terminated."),
                ("reset_{e}_credentials", "Reset credentials for a {E}. Forces re-authentication on all active sessions. Previous credentials are permanently invalidated."),
            ],
            "check": [
                ("audit_{e}_access", "Audit all access events for a {E}. Returns a detailed log of who accessed what and when."),
            ],
        },
    },
    "legal": {
        "label": "Contract Management / Legal",
        "entities": {
            "contract": {
                "id_param": ("contract_id", "Unique contract identifier"),
                "specific_params": {
                    "contract_type": {"type": "string", "enum": ["master_service_agreement", "nda", "sow", "license", "employment", "vendor"], "description": "Type of contract"},
                    "party_name": {"type": "string", "description": "Name of the counterparty to the contract"},
                    "effective_date": {"type": "string", "description": "Date the contract takes effect (YYYY-MM-DD)"},
                    "expiry_date": {"type": "string", "description": "Date the contract expires (YYYY-MM-DD)"},
                    "jurisdiction": {"type": "string", "description": "Governing law jurisdiction (e.g., State of Delaware, England and Wales)"},
                    "status": {"type": "string", "enum": ["draft", "active", "expired", "terminated"], "description": "Current contract status"},
                },
            },
            "clause": {
                "id_param": ("clause_id", "Unique clause identifier"),
                "specific_params": {
                    "contract_id": {"type": "string", "description": "Contract this clause belongs to"},
                    "clause_type": {"type": "string", "enum": ["indemnification", "limitation_of_liability", "termination", "confidentiality", "ip_ownership", "force_majeure", "governing_law"], "description": "Clause category"},
                    "section_number": {"type": "string", "description": "Section number within the contract (e.g., 4.2.1)"},
                    "text": {"type": "string", "description": "Full text of the clause"},
                    "risk_flag": {"type": "boolean", "description": "Whether this clause has been flagged for legal review"},
                },
            },
            "nda": {
                "id_param": ("nda_id", "Unique NDA identifier"),
                "specific_params": {
                    "party_name": {"type": "string", "description": "Counterparty to the NDA"},
                    "nda_type": {"type": "string", "enum": ["mutual", "unilateral"], "description": "Whether the NDA is mutual or one-way"},
                    "effective_date": {"type": "string", "description": "NDA effective date (YYYY-MM-DD)"},
                    "expiry_date": {"type": "string", "description": "NDA expiry date (YYYY-MM-DD)"},
                    "jurisdiction": {"type": "string", "description": "Governing jurisdiction"},
                    "status": {"type": "string", "enum": ["draft", "active", "expired", "terminated"], "description": "NDA status"},
                },
            },
            "legal_case": {
                "id_param": ("case_id", "Unique legal case identifier"),
                "specific_params": {
                    "case_title": {"type": "string", "description": "Case title or caption"},
                    "case_type": {"type": "string", "enum": ["litigation", "arbitration", "regulatory", "internal_investigation"], "description": "Type of legal proceeding"},
                    "status": {"type": "string", "enum": ["open", "discovery", "trial", "settled", "closed"], "description": "Current case status"},
                    "opposing_party": {"type": "string", "description": "Name of the opposing party"},
                    "assigned_attorney": {"type": "string", "description": "Lead attorney handling the case"},
                    "court_jurisdiction": {"type": "string", "description": "Court or tribunal jurisdiction"},
                },
            },
            "amendment": {
                "id_param": ("amendment_id", "Unique amendment identifier"),
                "specific_params": {
                    "contract_id": {"type": "string", "description": "Contract being amended"},
                    "amendment_number": {"type": "integer", "description": "Sequential amendment number"},
                    "effective_date": {"type": "string", "description": "Date the amendment takes effect (YYYY-MM-DD)"},
                    "description": {"type": "string", "description": "Summary of the changes made by this amendment"},
                    "status": {"type": "string", "enum": ["draft", "pending_approval", "executed", "rejected"], "description": "Amendment approval status"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve full details of a {E} including version history, related parties, and associated documents."),
                ("search_{e}s", "Search {E}s by party name, type, status, or date range. Returns matching records with key terms highlighted."),
                ("list_{e}s", "List all {E}s with optional filtering by status, jurisdiction, or expiry date. Includes summary metadata."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} record with the specified details. Validates required fields and jurisdiction-specific rules."),
                ("update_{e}", "Update an existing {E}'s fields. Changes are versioned and logged for audit trail compliance."),
                ("validate_{e}_compliance", "Run compliance checks on a {E} against organizational legal policies. Returns a list of issues or warnings."),
            ],
            "dangerous": [
                ("terminate_{e}", "Terminate a {E} before its natural expiry. Triggers counterparty notification and generates the termination notice. This action may have legal consequences."),
                ("archive_{e}", "Archive a {E}, removing it from active views. Archived records are retained per the document retention policy and can be restored."),
            ],
            "check": [
                ("check_{e}_expiry", "Check expiry status and upcoming renewal deadlines for a {E}. Returns days remaining and renewal recommendations."),
            ],
        },
    },
    "recruitment": {
        "label": "Recruitment / ATS",
        "entities": {
            "candidate": {
                "id_param": ("candidate_id", "Unique candidate identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Candidate full name"},
                    "email": {"type": "string", "description": "Candidate email address"},
                    "position_id": {"type": "string", "description": "Position the candidate is applying for"},
                    "experience_years": {"type": "integer", "description": "Total years of relevant professional experience"},
                    "stage": {"type": "string", "enum": ["applied", "screening", "interview", "offer", "hired", "rejected"], "description": "Current stage in the hiring pipeline"},
                    "source": {"type": "string", "enum": ["referral", "linkedin", "job_board", "careers_page", "agency", "internal"], "description": "How the candidate was sourced"},
                },
            },
            "job_posting": {
                "id_param": ("posting_id", "Unique job posting identifier"),
                "specific_params": {
                    "title": {"type": "string", "description": "Job title"},
                    "department": {"type": "string", "description": "Hiring department"},
                    "salary_range": {"type": "string", "description": "Salary range for the position (e.g., $120K-$160K)"},
                    "location": {"type": "string", "description": "Office location or remote/hybrid"},
                    "status": {"type": "string", "enum": ["draft", "open", "closed", "on_hold", "filled"], "description": "Posting status"},
                    "headcount": {"type": "integer", "description": "Number of positions to fill"},
                },
            },
            "interview": {
                "id_param": ("interview_id", "Unique interview identifier"),
                "specific_params": {
                    "candidate_id": {"type": "string", "description": "Candidate being interviewed"},
                    "interviewer_id": {"type": "string", "description": "Employee conducting the interview"},
                    "interview_type": {"type": "string", "enum": ["phone_screen", "technical", "behavioral", "panel", "onsite", "hiring_manager"], "description": "Type of interview"},
                    "scheduled_time": {"type": "string", "description": "Interview date and time (ISO 8601)"},
                    "duration_minutes": {"type": "integer", "description": "Expected interview duration in minutes"},
                    "status": {"type": "string", "enum": ["scheduled", "completed", "canceled", "no_show"], "description": "Interview status"},
                },
            },
            "application": {
                "id_param": ("application_id", "Unique application identifier"),
                "specific_params": {
                    "candidate_id": {"type": "string", "description": "Candidate who submitted the application"},
                    "position_id": {"type": "string", "description": "Position applied for"},
                    "resume_url": {"type": "string", "description": "URL to the candidate's resume"},
                    "cover_letter": {"type": "string", "description": "Cover letter text"},
                    "submitted_date": {"type": "string", "description": "Date the application was submitted (YYYY-MM-DD)"},
                    "status": {"type": "string", "enum": ["received", "under_review", "shortlisted", "rejected", "withdrawn"], "description": "Application review status"},
                },
            },
            "offer": {
                "id_param": ("offer_id", "Unique offer identifier"),
                "specific_params": {
                    "candidate_id": {"type": "string", "description": "Candidate receiving the offer"},
                    "position_id": {"type": "string", "description": "Position the offer is for"},
                    "salary": {"type": "number", "description": "Annual base salary offered"},
                    "start_date": {"type": "string", "description": "Proposed start date (YYYY-MM-DD)"},
                    "expiry_date": {"type": "string", "description": "Date the offer expires (YYYY-MM-DD)"},
                    "status": {"type": "string", "enum": ["draft", "sent", "accepted", "declined", "expired", "rescinded"], "description": "Offer status"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve full details of a {E} including history, notes, and associated records."),
                ("search_{e}s", "Search {E}s by name, department, stage, or date range. Returns matching records with pipeline context."),
                ("list_{e}s", "List all {E}s with optional filtering by position, stage, or status. Includes pipeline statistics."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} record. Validates required fields and triggers the appropriate workflow notifications."),
                ("update_{e}", "Update an existing {E}'s fields. Changes are logged in the activity timeline."),
                ("schedule_{e}", "Schedule or reschedule a {E}. Sends calendar invitations to all participants."),
                ("check_{e}_eligibility", "Check whether a {E} meets the minimum requirements for the associated position. Returns pass/fail with details."),
            ],
            "dangerous": [
                ("reject_{e}", "Reject a {E} and send the configured rejection notification. Updates the pipeline and frees the position slot. This action cannot be easily reversed."),
                ("withdraw_{e}", "Withdraw a {E} from the process at the candidate's request. Closes the associated application and notifies the hiring team."),
            ],
            "check": [
                ("check_{e}_pipeline", "Get pipeline metrics and stage distribution for a {E}. Returns time-in-stage, conversion rates, and bottleneck analysis."),
            ],
        },
    },
    "banking": {
        "label": "Banking / Financial Services",
        "entities": {
            "account": {
                "id_param": ("account_id", "Unique bank account identifier"),
                "specific_params": {
                    "account_number": {"type": "string", "description": "Bank account number"},
                    "routing_number": {"type": "string", "description": "Bank routing/sort code number"},
                    "account_type": {"type": "string", "enum": ["checking", "savings", "money_market", "cd", "business"], "description": "Type of bank account"},
                    "currency": {"type": "string", "enum": ["USD", "EUR", "GBP", "JPY", "CHF"], "description": "Account currency"},
                    "balance": {"type": "number", "description": "Current account balance"},
                    "status": {"type": "string", "enum": ["active", "dormant", "frozen", "closed"], "description": "Account status"},
                },
            },
            "transfer": {
                "id_param": ("transfer_id", "Unique transfer identifier"),
                "specific_params": {
                    "source_account_id": {"type": "string", "description": "Source account for the transfer"},
                    "destination_account_id": {"type": "string", "description": "Destination account for the transfer"},
                    "amount": {"type": "number", "description": "Transfer amount"},
                    "currency": {"type": "string", "description": "Transfer currency (ISO 4217)"},
                    "transaction_type": {"type": "string", "enum": ["credit", "debit", "transfer"], "description": "Type of transaction"},
                    "status": {"type": "string", "enum": ["pending", "processing", "completed", "failed", "reversed"], "description": "Transfer processing status"},
                    "reference": {"type": "string", "description": "Payment reference or memo"},
                },
            },
            "beneficiary": {
                "id_param": ("beneficiary_id", "Unique beneficiary identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Beneficiary name"},
                    "account_number": {"type": "string", "description": "Beneficiary account number"},
                    "routing_number": {"type": "string", "description": "Beneficiary bank routing number"},
                    "bank_name": {"type": "string", "description": "Beneficiary bank name"},
                    "country": {"type": "string", "description": "Beneficiary country (ISO 3166-1 alpha-2)"},
                    "risk_level": {"type": "string", "enum": ["low", "medium", "high"], "description": "Assessed risk level for this beneficiary"},
                },
            },
            "statement": {
                "id_param": ("statement_id", "Unique statement identifier"),
                "specific_params": {
                    "account_id": {"type": "string", "description": "Account this statement belongs to"},
                    "period_start": {"type": "string", "description": "Statement period start date (YYYY-MM-DD)"},
                    "period_end": {"type": "string", "description": "Statement period end date (YYYY-MM-DD)"},
                    "opening_balance": {"type": "number", "description": "Balance at the start of the period"},
                    "closing_balance": {"type": "number", "description": "Balance at the end of the period"},
                    "transaction_count": {"type": "integer", "description": "Number of transactions in the period"},
                },
            },
            "kyc_check": {
                "id_param": ("kyc_id", "Unique KYC check identifier"),
                "specific_params": {
                    "customer_id": {"type": "string", "description": "Customer being verified"},
                    "check_type": {"type": "string", "enum": ["identity", "address", "sanctions", "pep", "adverse_media"], "description": "Type of KYC check"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "passed", "failed", "requires_review"], "description": "KYC check status"},
                    "risk_level": {"type": "string", "enum": ["low", "medium", "high", "unacceptable"], "description": "Risk level determined by the check"},
                    "document_type": {"type": "string", "enum": ["passport", "drivers_license", "national_id", "utility_bill", "bank_statement"], "description": "Type of supporting document"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve full details of a {E} including transaction history and compliance flags. All access is logged for regulatory compliance."),
                ("search_{e}s", "Search {E}s by account number, customer, amount range, or date. Returns matching records with risk indicators."),
                ("list_{e}s", "List all {E}s with optional filtering by status, type, or date range. Includes summary balances and risk flags."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} record. Validates regulatory requirements and triggers compliance checks before creation."),
                ("initiate_{e}", "Initiate a new {E} transaction. Validates funds availability, beneficiary details, and applies fraud screening rules."),
                ("verify_{e}", "Mark a {E} as verified after completing the required checks. Updates the compliance record accordingly."),
            ],
            "dangerous": [
                ("freeze_{e}", "Freeze a {E}, blocking all transactions. Requires compliance officer authorization and triggers regulatory notification. Cannot be reversed without a formal review."),
                ("close_{e}", "Permanently close a {E}. Disburses remaining balance to the designated account and generates final statements. This action cannot be undone."),
                ("audit_{e}_transaction", "Initiate a formal audit of a {E}'s transactions for suspected fraud or regulatory violations. Flags the account and notifies the compliance team."),
            ],
            "check": [
                ("check_{e}_risk", "Run a risk assessment on a {E}. Returns risk score, suspicious activity indicators, and recommended actions."),
            ],
        },
    },
    "insurance": {
        "label": "Insurance",
        "entities": {
            "policy": {
                "id_param": ("policy_id", "Unique insurance policy identifier"),
                "specific_params": {
                    "policy_number": {"type": "string", "description": "Human-readable policy number"},
                    "policyholder_id": {"type": "string", "description": "Policyholder customer identifier"},
                    "policy_type": {"type": "string", "enum": ["auto", "health", "property", "life", "liability", "umbrella"], "description": "Type of insurance policy"},
                    "coverage_amount": {"type": "number", "description": "Total coverage amount in the policy currency"},
                    "deductible": {"type": "number", "description": "Deductible amount the policyholder must pay before coverage applies"},
                    "status": {"type": "string", "enum": ["active", "lapsed", "canceled", "expired", "pending_renewal"], "description": "Current policy status"},
                },
            },
            "claim": {
                "id_param": ("claim_id", "Unique claim identifier"),
                "specific_params": {
                    "policy_id": {"type": "string", "description": "Policy under which the claim is filed"},
                    "policyholder_id": {"type": "string", "description": "Policyholder filing the claim"},
                    "claim_type": {"type": "string", "enum": ["auto", "health", "property", "life"], "description": "Type of insurance claim"},
                    "incident_date": {"type": "string", "description": "Date the covered incident occurred (YYYY-MM-DD)"},
                    "claimed_amount": {"type": "number", "description": "Amount claimed by the policyholder"},
                    "status": {"type": "string", "enum": ["filed", "under_review", "approved", "denied", "settled", "appealed"], "description": "Claim processing status"},
                },
            },
            "underwriting_assessment": {
                "id_param": ("assessment_id", "Unique underwriting assessment identifier"),
                "specific_params": {
                    "applicant_id": {"type": "string", "description": "Applicant being assessed"},
                    "policy_type": {"type": "string", "enum": ["auto", "health", "property", "life"], "description": "Type of policy being underwritten"},
                    "risk_score": {"type": "number", "description": "Calculated risk score (0-100)"},
                    "risk_factors": {"type": "array", "items": {"type": "string"}, "description": "List of identified risk factors"},
                    "recommendation": {"type": "string", "enum": ["approve", "approve_with_conditions", "decline", "refer_to_senior"], "description": "Underwriting recommendation"},
                    "status": {"type": "string", "enum": ["pending", "in_progress", "completed"], "description": "Assessment status"},
                },
            },
            "premium": {
                "id_param": ("premium_id", "Unique premium payment identifier"),
                "specific_params": {
                    "policy_id": {"type": "string", "description": "Policy this premium payment is for"},
                    "amount": {"type": "number", "description": "Premium amount due"},
                    "billing_period": {"type": "string", "description": "Billing period covered by this premium (e.g., Jan 2026)"},
                    "due_date": {"type": "string", "description": "Payment due date (YYYY-MM-DD)"},
                    "status": {"type": "string", "enum": ["pending", "paid", "overdue", "waived"], "description": "Payment status"},
                },
            },
            "coverage": {
                "id_param": ("coverage_id", "Unique coverage line item identifier"),
                "specific_params": {
                    "policy_id": {"type": "string", "description": "Policy this coverage belongs to"},
                    "coverage_type": {"type": "string", "description": "Type of coverage (e.g., collision, comprehensive, bodily_injury)"},
                    "coverage_amount": {"type": "number", "description": "Maximum coverage amount for this line item"},
                    "deductible": {"type": "number", "description": "Deductible specific to this coverage line"},
                    "effective_date": {"type": "string", "description": "Date this coverage takes effect (YYYY-MM-DD)"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve full details of a {E} including coverage breakdown, payment history, and claim records."),
                ("search_{e}s", "Search {E}s by policyholder, policy number, type, or status. Returns matching records with risk context."),
                ("list_{e}s", "List all {E}s with optional filtering by type, status, or date range. Includes summary statistics."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} record. Validates coverage requirements and triggers underwriting if applicable."),
                ("update_{e}", "Update an existing {E}'s fields. Premium recalculation is triggered if coverage or risk factors change."),
                ("file_{e}", "File a new insurance {E}. Initiates the claims workflow and assigns an adjuster based on claim type and severity."),
                ("calculate_{e}_premium", "Calculate or recalculate the premium for a {E} based on current risk factors and coverage selections. Returns the updated premium breakdown."),
            ],
            "dangerous": [
                ("approve_{e}", "Approve a {E} for payout or issuance. Triggers disbursement or policy activation. Once approved, this decision is binding per regulatory requirements."),
                ("deny_{e}", "Deny a {E}. Sends the denial notice to the policyholder with the reason and appeal instructions. Denial decisions are subject to regulatory review."),
            ],
            "check": [
                ("check_{e}_eligibility", "Check eligibility and coverage limits for a {E}. Returns applicable deductibles, coverage caps, and exclusions."),
            ],
        },
    },
    "real_estate": {
        "label": "Real Estate / Property Management",
        "entities": {
            "property": {
                "id_param": ("property_id", "Unique property identifier"),
                "specific_params": {
                    "address": {"type": "string", "description": "Full property address"},
                    "property_type": {"type": "string", "enum": ["residential", "commercial", "industrial"], "description": "Property type classification"},
                    "square_footage": {"type": "number", "description": "Total square footage of the property"},
                    "unit_count": {"type": "integer", "description": "Number of rentable units in the property"},
                    "market_value": {"type": "number", "description": "Current estimated market value in USD"},
                    "status": {"type": "string", "enum": ["available", "occupied", "under_maintenance", "off_market"], "description": "Current property status"},
                },
            },
            "lease": {
                "id_param": ("lease_id", "Unique lease agreement identifier"),
                "specific_params": {
                    "property_id": {"type": "string", "description": "Property this lease is for"},
                    "tenant_id": {"type": "string", "description": "Tenant on the lease"},
                    "unit_number": {"type": "string", "description": "Specific unit within the property"},
                    "rent_amount": {"type": "number", "description": "Monthly rent amount"},
                    "lease_term": {"type": "string", "enum": ["month_to_month", "6_months", "12_months", "24_months", "36_months"], "description": "Lease duration"},
                    "start_date": {"type": "string", "description": "Lease start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "Lease end date (YYYY-MM-DD)"},
                    "status": {"type": "string", "enum": ["active", "expired", "terminated", "pending_renewal"], "description": "Lease status"},
                },
            },
            "tenant": {
                "id_param": ("tenant_id", "Unique tenant identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Tenant full name or business name"},
                    "email": {"type": "string", "description": "Tenant contact email"},
                    "phone": {"type": "string", "description": "Tenant contact phone number"},
                    "lease_id": {"type": "string", "description": "Current active lease identifier"},
                    "balance_due": {"type": "number", "description": "Outstanding balance due from the tenant"},
                    "status": {"type": "string", "enum": ["active", "past_due", "eviction_notice", "vacated"], "description": "Tenant account status"},
                },
            },
            "inspection": {
                "id_param": ("inspection_id", "Unique inspection identifier"),
                "specific_params": {
                    "property_id": {"type": "string", "description": "Property being inspected"},
                    "unit_number": {"type": "string", "description": "Specific unit inspected (if applicable)"},
                    "inspection_type": {"type": "string", "enum": ["move_in", "move_out", "routine", "maintenance", "safety"], "description": "Type of inspection"},
                    "inspector_id": {"type": "string", "description": "Inspector assigned to this inspection"},
                    "scheduled_date": {"type": "string", "description": "Scheduled inspection date (YYYY-MM-DD)"},
                    "status": {"type": "string", "enum": ["scheduled", "completed", "canceled", "overdue"], "description": "Inspection status"},
                },
            },
            "listing": {
                "id_param": ("listing_id", "Unique property listing identifier"),
                "specific_params": {
                    "property_id": {"type": "string", "description": "Property being listed"},
                    "listing_type": {"type": "string", "enum": ["sale", "lease", "sublease"], "description": "Type of listing"},
                    "asking_price": {"type": "number", "description": "Asking price or monthly rent"},
                    "description": {"type": "string", "description": "Marketing description for the listing"},
                    "photos_count": {"type": "integer", "description": "Number of photos attached to the listing"},
                    "status": {"type": "string", "enum": ["draft", "active", "under_contract", "sold", "leased", "delisted"], "description": "Listing status"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve full details of a {E} including financial history, associated leases, and maintenance records."),
                ("search_{e}s", "Search {E}s by address, type, status, or price range. Returns matching records with occupancy data."),
                ("list_{e}s", "List all {E}s with optional filtering by property, status, or date range. Includes portfolio summary metrics."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} record with the specified details. Validates property references and required fields."),
                ("update_{e}", "Update an existing {E}'s fields. Changes are logged in the property management audit trail."),
                ("schedule_{e}_inspection", "Schedule a new inspection for a {E}. Sends notifications to the inspector and tenant if applicable."),
                ("publish_{e}", "Publish a {E} to configured listing platforms and portals. Makes the listing publicly visible."),
            ],
            "dangerous": [
                ("terminate_{e}_lease", "Terminate a lease agreement before its natural end date. Triggers notice period calculations, security deposit processing, and tenant notification. May have legal implications."),
                ("delist_{e}", "Remove a {E} from all listing platforms immediately. Cancels any scheduled showings and notifies interested parties."),
            ],
            "check": [
                ("check_{e}_availability", "Check current availability and scheduling conflicts for a {E}. Returns vacancy status and upcoming lease expirations."),
            ],
        },
    },
    "retail": {
        "label": "Retail / E-commerce",
        "entities": {
            "product": {
                "id_param": ("product_id", "Unique product identifier"),
                "specific_params": {
                    "sku": {"type": "string", "description": "Stock keeping unit code"},
                    "product_name": {"type": "string", "description": "Product display name"},
                    "price": {"type": "number", "description": "Current retail price"},
                    "category": {"type": "string", "description": "Product category (e.g., electronics, apparel, home)"},
                    "quantity": {"type": "integer", "description": "Current inventory quantity across all warehouses"},
                    "status": {"type": "string", "enum": ["active", "discontinued", "out_of_stock", "pre_order"], "description": "Product availability status"},
                },
            },
            "order": {
                "id_param": ("order_id", "Unique order identifier"),
                "specific_params": {
                    "customer_id": {"type": "string", "description": "Customer who placed the order"},
                    "order_status": {"type": "string", "enum": ["pending", "processing", "shipped", "delivered", "cancelled"], "description": "Current order fulfillment status"},
                    "total_amount": {"type": "number", "description": "Total order amount including tax and shipping"},
                    "shipping_address": {"type": "string", "description": "Delivery address for the order"},
                    "tracking_number": {"type": "string", "description": "Shipping carrier tracking number"},
                    "placed_date": {"type": "string", "description": "Date the order was placed (YYYY-MM-DD)"},
                },
            },
            "cart": {
                "id_param": ("cart_id", "Unique shopping cart identifier"),
                "specific_params": {
                    "customer_id": {"type": "string", "description": "Customer who owns this cart"},
                    "items": {"type": "array", "items": {"type": "object"}, "description": "List of cart items with product IDs and quantities"},
                    "subtotal": {"type": "number", "description": "Cart subtotal before tax and shipping"},
                    "currency": {"type": "string", "description": "Cart currency code"},
                    "status": {"type": "string", "enum": ["active", "abandoned", "converted", "expired"], "description": "Cart status"},
                },
            },
            "return": {
                "id_param": ("return_id", "Unique return/RMA identifier"),
                "specific_params": {
                    "order_id": {"type": "string", "description": "Original order associated with this return"},
                    "customer_id": {"type": "string", "description": "Customer requesting the return"},
                    "reason": {"type": "string", "enum": ["defective", "wrong_item", "not_as_described", "changed_mind", "arrived_late", "other"], "description": "Reason for the return"},
                    "refund_amount": {"type": "number", "description": "Refund amount to process"},
                    "status": {"type": "string", "enum": ["requested", "approved", "item_received", "refunded", "denied"], "description": "Return processing status"},
                },
            },
            "promotion": {
                "id_param": ("promotion_id", "Unique promotion identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Promotion name or title"},
                    "discount_type": {"type": "string", "enum": ["percentage", "fixed_amount", "buy_one_get_one", "free_shipping"], "description": "Type of discount"},
                    "discount_value": {"type": "number", "description": "Discount value (percentage or fixed amount)"},
                    "promo_code": {"type": "string", "description": "Coupon or promo code customers enter at checkout"},
                    "start_date": {"type": "string", "description": "Promotion start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "Promotion end date (YYYY-MM-DD)"},
                    "status": {"type": "string", "enum": ["draft", "active", "scheduled", "expired", "disabled"], "description": "Promotion status"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve full details of a {E} including pricing history, inventory levels, and associated records."),
                ("search_{e}s", "Search {E}s by name, SKU, category, status, or price range. Returns matching results with inventory data."),
                ("list_{e}s", "List all {E}s with optional filtering by category, status, or date range. Includes aggregate metrics."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} record. Validates product references, pricing rules, and inventory availability."),
                ("update_{e}", "Update an existing {E}'s fields. Price changes trigger notification to subscribed customers."),
                ("apply_{e}_discount", "Apply a promotion or discount to a {E}. Validates eligibility rules and stacking policies before applying."),
                ("update_{e}_inventory", "Update inventory quantities for a {E}. Triggers reorder alerts if stock falls below the configured threshold."),
            ],
            "dangerous": [
                ("cancel_{e}_order", "Cancel an order before it ships. Triggers refund processing, restocks reserved inventory, and notifies the customer. Cannot be undone once refund is issued."),
                ("process_{e}_return", "Process a product return and initiate the refund. Updates inventory, processes the refund to the original payment method, and closes the RMA."),
            ],
            "check": [
                ("check_{e}_availability", "Check real-time inventory availability for a {E} across all fulfillment locations. Returns stock levels and estimated restock dates."),
            ],
        },
    },
    "education": {
        "label": "Education / LMS",
        "entities": {
            "course": {
                "id_param": ("course_id", "Unique course identifier"),
                "specific_params": {
                    "title": {"type": "string", "description": "Course title"},
                    "department": {"type": "string", "description": "Academic department offering the course"},
                    "instructor_id": {"type": "string", "description": "Primary instructor's faculty ID"},
                    "credit_hours": {"type": "integer", "description": "Number of credit hours the course carries"},
                    "semester": {"type": "string", "description": "Academic semester (e.g., Fall 2026, Spring 2027)"},
                    "capacity": {"type": "integer", "description": "Maximum enrollment capacity"},
                    "status": {"type": "string", "enum": ["draft", "open_enrollment", "in_progress", "completed", "canceled"], "description": "Course offering status"},
                },
            },
            "enrollment": {
                "id_param": ("enrollment_id", "Unique enrollment record identifier"),
                "specific_params": {
                    "student_id": {"type": "string", "description": "Student enrolled in the course"},
                    "course_id": {"type": "string", "description": "Course the student is enrolled in"},
                    "semester": {"type": "string", "description": "Academic semester of enrollment"},
                    "enrollment_status": {"type": "string", "enum": ["active", "completed", "withdrawn", "incomplete"], "description": "Current enrollment status"},
                    "enrolled_date": {"type": "string", "description": "Date the student enrolled (YYYY-MM-DD)"},
                    "final_grade": {"type": "string", "description": "Final letter grade (if course is completed)"},
                },
            },
            "student": {
                "id_param": ("student_id", "Unique student identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Student full name"},
                    "email": {"type": "string", "description": "Student email address"},
                    "program": {"type": "string", "description": "Degree program or major"},
                    "gpa": {"type": "number", "description": "Current cumulative GPA (0.0-4.0)"},
                    "enrollment_year": {"type": "integer", "description": "Year the student first enrolled"},
                    "status": {"type": "string", "enum": ["active", "on_leave", "graduated", "suspended", "withdrawn"], "description": "Student enrollment status"},
                },
            },
            "assignment": {
                "id_param": ("assignment_id", "Unique assignment identifier"),
                "specific_params": {
                    "course_id": {"type": "string", "description": "Course this assignment belongs to"},
                    "title": {"type": "string", "description": "Assignment title"},
                    "assignment_type": {"type": "string", "enum": ["homework", "quiz", "midterm", "final_exam", "project", "lab"], "description": "Type of assignment"},
                    "due_date": {"type": "string", "description": "Assignment due date (YYYY-MM-DD)"},
                    "max_points": {"type": "integer", "description": "Maximum points for this assignment"},
                    "status": {"type": "string", "enum": ["draft", "published", "closed", "graded"], "description": "Assignment status"},
                },
            },
            "grade": {
                "id_param": ("grade_id", "Unique grade record identifier"),
                "specific_params": {
                    "student_id": {"type": "string", "description": "Student who received the grade"},
                    "assignment_id": {"type": "string", "description": "Assignment being graded"},
                    "course_id": {"type": "string", "description": "Course the grade belongs to"},
                    "grade_value": {"type": "number", "description": "Numeric grade or score (e.g., 87.5)"},
                    "letter_grade": {"type": "string", "enum": ["A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F", "I", "W"], "description": "Letter grade equivalent"},
                    "graded_date": {"type": "string", "description": "Date the grade was submitted (YYYY-MM-DD)"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve full details of a {E} including enrollment data, grades, and associated records."),
                ("search_{e}s", "Search {E}s by name, course, semester, or status. Returns matching records with academic context."),
                ("list_{e}s", "List all {E}s with optional filtering by course, semester, or status. Includes enrollment counts and averages."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} record. Validates academic references and prerequisite requirements before saving."),
                ("update_{e}", "Update an existing {E}'s fields. Changes to grades require instructor authorization."),
                ("enroll_{e}", "Enroll a student in a course. Validates prerequisites, capacity limits, and schedule conflicts before confirming."),
                ("submit_{e}_grade", "Submit or update a grade for a {E}. Validates against the assignment rubric and notifies the student."),
            ],
            "dangerous": [
                ("withdraw_{e}", "Withdraw a student from a course. Applies the institution's withdrawal policy for grading (W grade) and refund eligibility. Deadline-dependent and cannot be reversed after the drop period."),
                ("check_{e}_prerequisite", "Check whether prerequisite requirements are met for a {E}. Returns pass/fail with details on missing requirements."),
            ],
            "check": [
                ("check_{e}_standing", "Check academic standing and progress for a {E}. Returns GPA, credit hours completed, and eligibility alerts."),
            ],
        },
    },
    "logistics": {
        "label": "Logistics / Transportation",
        "entities": {
            "shipment": {
                "id_param": ("shipment_id", "Unique shipment identifier"),
                "specific_params": {
                    "tracking_number": {"type": "string", "description": "Carrier-assigned tracking number"},
                    "origin": {"type": "string", "description": "Origin facility or address"},
                    "destination": {"type": "string", "description": "Destination facility or address"},
                    "carrier": {"type": "string", "description": "Shipping carrier name"},
                    "weight_kg": {"type": "number", "description": "Total shipment weight in kilograms"},
                    "delivery_status": {"type": "string", "enum": ["pending", "in_transit", "out_for_delivery", "delivered", "failed"], "description": "Current delivery status"},
                },
            },
            "route": {
                "id_param": ("route_id", "Unique route identifier"),
                "specific_params": {
                    "origin": {"type": "string", "description": "Route starting point"},
                    "destination": {"type": "string", "description": "Route endpoint"},
                    "distance_km": {"type": "number", "description": "Total route distance in kilometers"},
                    "estimated_hours": {"type": "number", "description": "Estimated transit time in hours"},
                    "vehicle_id": {"type": "string", "description": "Vehicle assigned to this route"},
                    "status": {"type": "string", "enum": ["planned", "active", "completed", "canceled"], "description": "Route status"},
                },
            },
            "vehicle": {
                "id_param": ("vehicle_id", "Unique vehicle or fleet asset identifier"),
                "specific_params": {
                    "plate_number": {"type": "string", "description": "Vehicle license plate number"},
                    "vehicle_type": {"type": "string", "enum": ["van", "truck", "semi_trailer", "refrigerated", "flatbed"], "description": "Vehicle type classification"},
                    "capacity_kg": {"type": "number", "description": "Maximum load capacity in kilograms"},
                    "current_location": {"type": "string", "description": "Current GPS location or last known facility"},
                    "status": {"type": "string", "enum": ["available", "in_transit", "maintenance", "out_of_service"], "description": "Current vehicle status"},
                    "driver_id": {"type": "string", "description": "Driver currently assigned to this vehicle"},
                },
            },
            "warehouse": {
                "id_param": ("warehouse_id", "Unique warehouse facility identifier"),
                "specific_params": {
                    "name": {"type": "string", "description": "Warehouse facility name"},
                    "address": {"type": "string", "description": "Warehouse physical address"},
                    "capacity_sqft": {"type": "number", "description": "Total storage capacity in square feet"},
                    "utilization_pct": {"type": "number", "description": "Current space utilization percentage (0-100)"},
                    "warehouse_type": {"type": "string", "enum": ["general", "cold_storage", "hazmat", "bonded", "cross_dock"], "description": "Warehouse type classification"},
                    "status": {"type": "string", "enum": ["operational", "maintenance", "closed"], "description": "Facility operational status"},
                },
            },
            "delivery": {
                "id_param": ("delivery_id", "Unique delivery attempt identifier"),
                "specific_params": {
                    "shipment_id": {"type": "string", "description": "Shipment this delivery belongs to"},
                    "recipient_name": {"type": "string", "description": "Recipient name for the delivery"},
                    "delivery_address": {"type": "string", "description": "Delivery address"},
                    "scheduled_date": {"type": "string", "description": "Scheduled delivery date (YYYY-MM-DD)"},
                    "time_window": {"type": "string", "enum": ["morning", "afternoon", "evening", "all_day"], "description": "Preferred delivery time window"},
                    "status": {"type": "string", "enum": ["scheduled", "out_for_delivery", "delivered", "failed", "rescheduled"], "description": "Delivery attempt status"},
                    "signature_required": {"type": "boolean", "description": "Whether a signature is required upon delivery"},
                },
            },
        },
        "operations": {
            "read": [
                ("get_{e}", "Retrieve full details of a {E} including tracking events, route data, and proof of delivery."),
                ("search_{e}s", "Search {E}s by tracking number, origin, destination, carrier, or status. Returns matching records with logistics context."),
                ("list_{e}s", "List all {E}s with optional filtering by status, carrier, or date range. Includes fleet and capacity summaries."),
            ],
            "write": [
                ("create_{e}", "Create a new {E} record. Validates capacity, route feasibility, and carrier availability before confirmation."),
                ("update_{e}", "Update an existing {E}'s details. Triggers downstream notifications if delivery dates or destinations change."),
                ("dispatch_{e}", "Dispatch a {E} for pickup or delivery. Assigns a vehicle and driver, generates the route, and notifies the carrier."),
                ("reroute_{e}", "Reroute a {E} to a different destination or via an alternative path. Recalculates ETA and notifies all affected parties."),
            ],
            "dangerous": [
                ("cancel_{e}_shipment", "Cancel a shipment. Reverses carrier bookings, releases reserved capacity, and notifies the sender and recipient. Cannot be undone once the carrier has picked up the goods."),
                ("check_{e}_capacity", "Check available capacity at a warehouse or on a route. Returns current utilization, available slots, and upcoming reservations."),
            ],
            "check": [
                ("track_{e}", "Get real-time tracking for a {E}. Returns current location, estimated arrival, and full event history."),
            ],
        },
    },
}

# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------

def _entity_label(entity_key: str) -> str:
    """Convert entity_key like 'purchase_order' to 'purchase order'."""
    return entity_key.replace("_", " ")


def _pick_params(entity_cfg: dict, op_type: str, op_template_name: str) -> tuple:
    """
    Choose appropriate parameters for an operation.
    Returns (properties_dict, required_list).
    """
    id_name, id_desc = entity_cfg["id_param"]
    specific = entity_cfg["specific_params"]
    props = {}
    required = []

    if op_type == "read":
        if "get_" in op_template_name:
            # Get by ID: just the ID param
            props[id_name] = {"type": "string", "description": id_desc}
            required.append(id_name)
            props["fields"] = COMMON_PARAMS["fields"]
        elif "search_" in op_template_name or "list_" in op_template_name:
            # Search/list: pick 2-3 filterable params + pagination
            filterable = [k for k, v in specific.items()
                          if v.get("type") in ("string", "integer", "number")
                          and k != id_name]
            for p in filterable[:3]:
                props[p] = dict(specific[p])
            props["limit"] = COMMON_PARAMS["limit"]
            props["offset"] = COMMON_PARAMS["offset"]
            if "date" not in " ".join(filterable[:3]):
                props["date_from"] = COMMON_PARAMS["date_from"]
                props["date_to"] = COMMON_PARAMS["date_to"]
        elif "run_" in op_template_name or "track_" in op_template_name or "download_" in op_template_name:
            # Run/track/download: by ID
            props[id_name] = {"type": "string", "description": id_desc}
            required.append(id_name)
            if "run_" in op_template_name:
                props["dry_run"] = COMMON_PARAMS["dry_run"]

    elif op_type == "write":
        if "create_" in op_template_name or "upload_" in op_template_name:
            # Create: most specific params are included, top 4-5 required
            for k, v in specific.items():
                props[k] = dict(v)
            # First 2-3 specific params are required
            req_candidates = [k for k in specific.keys()
                              if specific[k].get("type") not in ("array", "object", "boolean")]
            required.extend(req_candidates[:3])
            props["tags"] = COMMON_PARAMS["tags"]
        elif "update_" in op_template_name:
            props[id_name] = {"type": "string", "description": id_desc}
            required.append(id_name)
            # Allow updating 3-4 fields
            updatable = list(specific.keys())[:4]
            for k in updatable:
                props[k] = dict(specific[k])
        elif "assign_" in op_template_name or "share_" in op_template_name:
            props[id_name] = {"type": "string", "description": id_desc}
            required.append(id_name)
            # Add an assignee/target param
            if "assignee_id" in specific:
                props["assignee_id"] = dict(specific["assignee_id"])
                required.append("assignee_id")
            elif "owner_id" in specific:
                props["owner_id"] = dict(specific["owner_id"])
                required.append("owner_id")
            else:
                props["target_user_id"] = {"type": "string", "description": "User or group to assign to"}
                required.append("target_user_id")
            props["notify"] = COMMON_PARAMS["notify"]
        elif "approve_" in op_template_name or "launch_" in op_template_name or "schedule_" in op_template_name:
            props[id_name] = {"type": "string", "description": id_desc}
            required.append(id_name)
            props["notes"] = COMMON_PARAMS["notes"]
        elif "send_" in op_template_name:
            for k, v in specific.items():
                props[k] = dict(v)
            req_candidates = [k for k in specific.keys()
                              if specific[k].get("type") not in ("array", "object", "boolean")]
            required.extend(req_candidates[:3])
        elif "escalate_" in op_template_name:
            props[id_name] = {"type": "string", "description": id_desc}
            required.append(id_name)
            props["target_tier"] = {"type": "string", "enum": ["tier_2", "tier_3", "manager"], "description": "Escalation target tier"}
            required.append("target_tier")
            props["reason"] = COMMON_PARAMS["reason"]
            required.append("reason")

    elif op_type == "dangerous":
        props[id_name] = {"type": "string", "description": id_desc}
        required.append(id_name)
        props["reason"] = COMMON_PARAMS["reason"]
        required.append("reason")
        # Some dangerous ops have confirmation
        if any(kw in op_template_name for kw in ["purge_", "delete_", "reset_", "force_close_"]):
            props["confirm"] = {"type": "boolean", "description": "Must be set to true to confirm this destructive action"}
            required.append("confirm")
        # Bulk operations get a filter
        if "bulk_" in op_template_name:
            props["filter_criteria"] = {"type": "object", "description": "Criteria to match records for bulk operation"}
            required.append("filter_criteria")

    elif op_type == "check":
        props[id_name] = {"type": "string", "description": id_desc}
        required.append(id_name)
        props["include_archived"] = COMMON_PARAMS["include_archived"]

    return props, required


def generate_schemas() -> list:
    """Generate all tool schemas from domain definitions."""
    all_schemas = []
    used_names = set()

    for domain_key, domain_cfg in DOMAINS.items():
        for entity_key, entity_cfg in domain_cfg["entities"].items():
            e_label = _entity_label(entity_key)
            for op_type, ops in domain_cfg["operations"].items():
                for op_template_name, op_template_desc in ops:
                    # Build tool name: replace {e}s with pluralized, {e} with singular
                    tool_name = op_template_name.replace("{e}s", pluralize(entity_key)).replace("{e}", entity_key)

                    # Skip if this name was already generated (e.g., from entity overlap)
                    if tool_name in used_names:
                        # Disambiguate by prepending domain
                        tool_name = f"{domain_key}_{tool_name}"
                    if tool_name in used_names:
                        continue
                    used_names.add(tool_name)

                    # Build description: pluralize labels too
                    e_label_plural = pluralize(e_label)
                    desc = op_template_desc.replace("{E}s", e_label_plural).replace("{E}", e_label).replace("{e}s", e_label_plural).replace("{e}", e_label)

                    # Build parameters
                    properties, required = _pick_params(entity_cfg, op_type, op_template_name)

                    schema = {
                        "name": tool_name,
                        "description": desc,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                        "domain": domain_key,
                        "operation_type": op_type,
                        "source": "generated",
                    }
                    all_schemas.append(schema)

    return all_schemas


def split_train_held_out(schemas: list, held_out_per_domain: int = 3) -> tuple:
    """
    Split schemas into training and held-out sets.
    Takes `held_out_per_domain` schemas per domain for the held-out set,
    ensuring diverse operation types.
    """
    by_domain = defaultdict(list)
    for s in schemas:
        by_domain[s["domain"]].append(s)

    train = []
    held_out = []

    for domain_key, domain_schemas in by_domain.items():
        # Group by operation type within domain
        by_op = defaultdict(list)
        for s in domain_schemas:
            by_op[s["operation_type"]].append(s)

        # Pick one from each op type for held-out, then fill remaining from largest groups
        held_out_picks = []
        for op_type in ["read", "write", "dangerous", "check"]:
            if by_op[op_type]:
                pick = random.choice(by_op[op_type])
                held_out_picks.append(pick)
                by_op[op_type].remove(pick)
                if len(held_out_picks) >= held_out_per_domain:
                    break

        # If we still need more, take from the largest remaining group
        while len(held_out_picks) < held_out_per_domain:
            largest_type = max(by_op, key=lambda t: len(by_op[t]))
            if not by_op[largest_type]:
                break
            pick = random.choice(by_op[largest_type])
            held_out_picks.append(pick)
            by_op[largest_type].remove(pick)

        held_out_ids = {id(s) for s in held_out_picks}
        for s in domain_schemas:
            if id(s) in held_out_ids:
                held_out.append(s)
            else:
                train.append(s)

    return train, held_out


def print_summary(train: list, held_out: list):
    """Print a summary of the generated schemas."""
    print(f"\n{'='*70}")
    print(f"  Enterprise Tool Schema Library -- Generation Summary")
    print(f"{'='*70}\n")

    print(f"  Total schemas generated: {len(train) + len(held_out)}")
    print(f"  Training set:            {len(train)}")
    print(f"  Held-out set:            {len(held_out)}")
    print()

    # Per-domain breakdown
    print(f"  {'Domain':<25} {'Train':>7} {'Held':>7} {'Total':>7}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7}")

    domain_order = list(DOMAINS.keys())
    for dk in domain_order:
        label = DOMAINS[dk]["label"]
        t_count = sum(1 for s in train if s["domain"] == dk)
        e_count = sum(1 for s in held_out if s["domain"] == dk)
        print(f"  {label:<25} {t_count:>7} {e_count:>7} {t_count+e_count:>7}")

    print()

    # Operation type breakdown
    print(f"  {'Operation Type':<25} {'Train':>7} {'Held':>7} {'Total':>7} {'%':>6}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")

    total_all = len(train) + len(held_out)
    for op_type in ["read", "write", "dangerous", "check"]:
        t_count = sum(1 for s in train if s["operation_type"] == op_type)
        e_count = sum(1 for s in held_out if s["operation_type"] == op_type)
        total = t_count + e_count
        pct = (total / total_all * 100) if total_all else 0
        print(f"  {op_type:<25} {t_count:>7} {e_count:>7} {total:>7} {pct:>5.1f}%")

    print()

    # Name uniqueness check
    all_names = [s["name"] for s in train + held_out]
    unique_names = set(all_names)
    if len(all_names) == len(unique_names):
        print(f"  Name uniqueness:         All {len(all_names)} names are unique")
    else:
        dupes = len(all_names) - len(unique_names)
        print(f"  WARNING: {dupes} duplicate names found!")
    print()


def main():
    base_dir = Path(__file__).resolve().parent.parent
    out_dir = base_dir / "data" / "v3" / "tool_schemas"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate
    all_schemas = generate_schemas()

    # Split
    train, held_out = split_train_held_out(all_schemas, held_out_per_domain=3)

    # Clean for output (keep all fields for downstream use)
    def clean(s):
        return {
            "name": s["name"],
            "description": s["description"],
            "parameters": s["parameters"],
            "domain": s["domain"],
            "operation_type": s["operation_type"],
            "source": s["source"],
        }

    train_clean = [clean(s) for s in train]
    held_out_clean = [clean(s) for s in held_out]

    # Save
    train_path = out_dir / "schemas.json"
    held_out_path = out_dir / "held_out_schemas.json"

    with open(train_path, "w") as f:
        json.dump(train_clean, f, indent=2)
    print(f"  Saved: {train_path}")

    with open(held_out_path, "w") as f:
        json.dump(held_out_clean, f, indent=2)
    print(f"  Saved: {held_out_path}")

    # Summary
    print_summary(train, held_out)


if __name__ == "__main__":
    main()
