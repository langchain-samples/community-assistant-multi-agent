"""Mock external systems: knowledge base, support cases, training catalog."""

KB_CORPUS: list[dict] = [
    {
        "id": "KB-001",
        "title": "Configuring time off accrual for HCM",
        "url": "https://community.example.com/kb/KB-001",
        "entitlement": "HCM",
        "body": "Time off accrual in HCM is configured per Time Off Plan. Set the accrual "
        "frequency, carryover limit and eligibility rules, then run Process Time Off Accruals.",
    },
    {
        "id": "KB-002",
        "title": "Troubleshooting time off accrual balances",
        "url": "https://community.example.com/kb/KB-002",
        "entitlement": "HCM",
        "body": "If a worker's time off accrual balance looks wrong, check the plan's "
        "eligibility date, any pending adjustments and the last accrual run.",
    },
    {
        "id": "KB-003",
        "title": "Payroll calendars and pay period setup",
        "url": "https://community.example.com/kb/KB-003",
        "entitlement": "Payroll",
        "body": "Payroll calendars define pay periods and pay dates. Create the calendar "
        "before assigning workers to a pay group.",
    },
    {
        "id": "KB-004",
        "title": "Payroll time off accrual payout on termination",
        "url": "https://community.example.com/kb/KB-004",
        "entitlement": "Payroll",
        "body": "Accrued time off can be paid out at termination using a payout earning "
        "code. Payroll-only: requires the Payroll entitlement.",
    },
    {
        "id": "KB-005",
        "title": "Running a payroll reconciliation report",
        "url": "https://community.example.com/kb/KB-005",
        "entitlement": "Payroll",
        "body": "Use the reconciliation report to compare gross-to-net results between "
        "two payroll runs and investigate variances.",
    },
    {
        "id": "KB-006",
        "title": "Business process security basics",
        "url": "https://community.example.com/kb/KB-006",
        "entitlement": "HCM",
        "body": "Security groups control who can initiate and approve business processes. "
        "Review domain security policies before activating pending changes.",
    },
    {
        "id": "KB-007",
        "title": "Loading worker data with an integration template",
        "url": "https://community.example.com/kb/KB-007",
        "entitlement": "HCM",
        "body": "Use the worker import template to bulk load hires. Validate the file "
        "in a sandbox tenant before loading to production.",
    },
    {
        "id": "KB-008",
        "title": "Understanding earning and deduction codes",
        "url": "https://community.example.com/kb/KB-008",
        "entitlement": "Payroll",
        "body": "Earning and deduction codes drive how pay components are calculated and "
        "reported on the pay slip.",
    },
    {
        "id": "KB-009",
        "title": "Running an off-cycle payroll",
        "url": "https://community.example.com/kb/KB-009",
        "entitlement": "Payroll",
        "body": "Use an off-cycle payroll for a manual payment, an on-demand payment or a "
        "correction outside the regular pay run. Start the Run Pay Calculation task for an "
        "off-cycle pay run, choose the workers and the pay group, enter the earnings, then "
        "review the results and complete the off-cycle payroll before the settlement run.",
    },
]

CASES: dict[str, dict] = {
    "CS-1042": {
        "id": "CS-1042",
        "owner": "admin_customer",
        "status": "Waiting on Acme",
        "subject": "Time off accrual balances not updating after plan change",
        "description": "After the Time Off Plan change, accrual balances stopped updating "
        "for workers in the affected plan.",
        "product": "Acme HCM",
        "opened": "2026-10-21",
        "last_update": "Engineering is reviewing the accrual run logs for your tenant.",
    },
}

TRAINING_CATALOG: dict[str, list[dict]] = {
    "Acme HCM": [
        {"id": "TR-HCM-101", "title": "HCM Fundamentals", "format": "Self-paced", "duration": "3h"},
        {"id": "TR-HCM-210", "title": "Time Off and Absence Administration", "format": "Instructor-led", "duration": "1d"},
    ],
    "Acme Payroll": [
        {"id": "TR-PAY-101", "title": "Payroll Fundamentals", "format": "Self-paced", "duration": "4h"},
        {"id": "TR-PAY-220", "title": "Payroll Reconciliation Workshop", "format": "Instructor-led", "duration": "1d"},
    ],
}
