# PROMPT Result Analysis Tool - Python Streamlit
for 2GO, Brylle Nunez

A Streamlit app to automate the analysis of PROMPT results by 2GO IT Department.

## Requirements
```
"streamlit>=1.52.0",
"plotly>=5.22.0",
"numpy>=1.26.0",
"pyinstaller>=6.17.0",
```

For developers:
- must have file named `employee_names_fn_ln.csv` containing names of the employees to check for compliance.
- must have `exempted.csv` to allow for optional uploading of exempted employees file.

## Features
* Upload CSV file: a raw CSV file containing entries of the employees. Must have the following columns:

    * `task_date`
    * `hours_spent`
    * `employee`
    * `racfid`
    * `rank`
    * `unit`
    * `remarks`
    * `ticket_number`
    * `ticket_type`
    * `work_setup`
    * `service_type`
    * `type`
    * `project`
    * `leader_name`

* Upload exempted employees CSV file: a raw CSV file containing employees to be exempted.

## Stats Generated

* Compliance Rating
    * By Unit
    * Overall

* Manhour Utilization
* Manhours Spent by Work Setup
* Work Setup Share by Unit
* Manhours Spent by Task Type
* Manhours Distribution by Task Type
* Manhours Spent by Ticket Type
* SD Units Analysis
    * SD - Consumer Task Type Distribution
    * SD - Corporate Task Type Distribution
    * SD - Consumer Ticket Type Distribution
    * SD - Corporate Ticket Type Distribution
* Manhours Spent by Ticket Type - Units
* Top Projects by Manhours

## Milestones/TODO
* Enable multi-ranged holiday selection since some months have multiple holidays.
* Refactoring of the codebase to improve readability.