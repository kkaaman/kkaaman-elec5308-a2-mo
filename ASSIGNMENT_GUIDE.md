# Getting Started with ELEC5308 Assignment 1

## Use AI as a Planner, Not as the Project Owner

Assignment 1 is a team engineering project with several interconnected requirements. Before producing substantial code, your team should develop a shared understanding of:

- what the project must achieve;
- what is explicitly required by the assignment;
- what remains for your team to decide;
- how the system could be divided into modules;
- what a credible minimum working system would look like;
- how team members will contribute, review and integrate their work.

GitHub Copilot or another University's approved AI tool can help facilitate this planning process. However, the AI should not make the project decisions on behalf of your team.

> **Use AI to ask questions, expose assumptions and organise your decisions. Your team remains responsible for the master plan.**

---

## Outcomes for This Activity

By the end of this activity, your team should have:

1. forked the group repository into each member's personal GitHub account;
2. reviewed the Assignment 1 specification;
3. used AI to investigate the scope of the project;
4. discussed the important project decisions as a team;
5. created an initial project master plan;
6. reviewed and agreed on the master plan;
7. added the agreed plan to the group repository;
8. identified initial contributions that team members can begin through their personal forks.

---

# Part 1: Fork the Group Repository

Each team member should begin by forking the Assignment 1 group repository into their personal GitHub account.

This gives every student:

- a personal copy of the repository for experimentation;
- a clear location for their individual contributions;
- an opportunity to work without directly changing the shared group repository;
- a visible record of commits and development activity;
- a pathway for sharing work through pull requests.

## Step 1: Create your personal fork

On GitHub:

1. Open the Assignment 1 group repository.
2. Select **Fork**.
3. Choose your personal GitHub account as the destination.
4. Keep the repository name unless your team has agreed on another convention.
5. Create the fork.

## Step 2: Clone your personal fork

Clone your fork to your computer and open it in Visual Studio Code.

```bash
git clone https://github.com/YOUR-USERNAME/GROUP-REPOSITORY.git
cd GROUP-REPOSITORY
code .
```

Replace the URL and repository name with those of your personal fork.

## Step 3: Add the group repository as `upstream`

Your personal fork is normally called `origin`. Add the group repository as `upstream` so that you can retrieve changes made by the team.

```bash
git remote add upstream https://github.com/GROUP-ORGANISATION/GROUP-REPOSITORY.git
git remote -v
```

You should see both:

- `origin`, which points to your personal fork;
- `upstream`, which points to the shared group repository.

## Step 4: Keep your fork current

Before beginning new work, retrieve the latest changes from the group repository.

```bash
git fetch upstream
git switch main
git merge upstream/main
git push origin main
```

If your repository uses a default branch name other than `main`, use that branch name instead.

> Tip: Do not develop every feature directly on `main`. Create a focused branch for each contribution.

For example:

```bash
git switch -c planning/project-masterplan
```

---

# Part 2: Use AI to Understand the Project Scope

Open the Assignment 1 specification and your repository in Visual Studio Code. Ensure that your AI Journal is recording significant interactions.

Start with planning. Do not ask AI to generate the complete assignment.

Avoid a broad request such as:

```text
Complete ELEC5308 Assignment 1 for us.
```

Instead, ask AI to help your team identify requirements, decisions, risks and unanswered questions.

## Suggested approach: AI as a senior AI engineer

Ask the AI to role-play as a senior AI engineer who is joining the initial project meeting. Its role is to interview the team, not to answer on your behalf.

Paste and adapt the following prompt:

```text
Act as a senior AI and autonomous-systems engineer facilitating the first
planning meeting for our ELEC5308 Assignment 1 team.

Read the Assignment 1 specification available in this workspace.

Your task is to interview us so that we can understand and define the scope of
our project.

Interview rules:

1. Ask one focused question at a time.
2. Wait for our response before continuing.
3. Do not answer questions on our behalf.
4. Challenge responses that are vague, contradictory, too broad or difficult
   to verify.
5. Distinguish explicit assignment requirements from team decisions and your
   own recommendations.
6. Ask follow-up questions when important assumptions are unclear.
7. Preserve disagreements and unresolved questions instead of silently making
   decisions for us.
8. Do not generate implementation code during this interview.
9. Do not select our simulator, architecture, algorithms or evaluation methods
   unless we explicitly make and justify those decisions.
10. After the interview, summarise our agreed decisions, provisional decisions,
    risks and unresolved questions.

The interview should cover:

- the purpose and intended demonstration of the project;
- the required perception and navigation capabilities;
- the minimum credible end-to-end baseline;
- simulator-independent architecture;
- possible modules and their responsibilities;
- interfaces and information exchanged between modules;
- simulator adapter responsibilities;
- testing and evaluation evidence;
- documentation and reproducibility;
- team workflow, code review and integration;
- responsible use of AI;
- major risks, dependencies and unresolved decisions.

Begin by explaining the process briefly, then ask the first question.
```

---

# Part 3: Discuss Before Responding

The educational value of this activity comes from the team discussion, not from the AI's final summary.

For every substantial question:

1. allow each member to express an opinion;
2. identify points of agreement and disagreement;
3. check the assignment specification before accepting claims about requirements;
4. give the AI the team's agreed response;
5. record unresolved matters when agreement has not yet been reached.

One group member may operate Copilot, but no single person should answer every question without consulting the team.

## Suggested rotating roles

- **Facilitator:** keeps the discussion focused on the current question.
- **Specification checker:** verifies claims against the assignment document.
- **Sceptic:** challenges assumptions and asks what evidence would be needed.
- **Recorder:** records agreements, disagreements and unresolved decisions.
- **Operator:** enters the team's agreed response into Copilot Chat.

Adapt the roles to your group size.

> Did you know that you can ask AI to keep track of the notes and writes them directly into a file? Search for it or ask AI how to do it!

---

# Alternative: Individual Interviews Followed by Comparison

Your team may also complete the AI interview individually before meeting as a group.

This approach can reveal different interpretations that might otherwise remain hidden.

## Individual phase

Each student should:

1. run the interview using their personal repository fork;
2. respond independently based on their understanding of the assignment;
3. ask the AI to summarise their proposed scope and priorities;
4. save the result in a temporary planning file or bring it to the group discussion;
5. record the interaction in their AI Journal.

## Group comparison phase

Compare the individual responses and identify:

- requirements that everyone interpreted consistently;
- different definitions of the minimum working system;
- alternative architectural ideas;
- different assumptions about module boundaries;
- disagreements about simulator choice or technical methods;
- risks noticed by only one team member;
- decisions that require evidence before the team commits to them.

Do not select an idea merely because AI described it confidently. Compare the reasoning, evidence, feasibility and alignment with the assignment.

A useful comparison prompt is:

```text
We will provide several individual interpretations of the Assignment 1 scope.

Compare them without choosing a winner.

Identify:

1. points of agreement;
2. important differences;
3. conflicting assumptions;
4. requirements that may have been misunderstood;
5. ideas that need evidence or a small experiment;
6. decisions the team must make together.

Do not merge the responses into a final plan until we discuss these issues.
```

---

# Part 4: Ask AI to Draft the Project Master Plan

After the interview and team discussion, ask AI to draft a master plan based on the team's responses.

The plan should be saved as:

```text
docs/project-masterplan.md
```

Use the following prompt:

```text
Using the Assignment 1 specification and our responses during the interview,
draft `docs/project-masterplan.md`.

The master plan must contain:

1. Project purpose
2. Explicit Assignment 1 requirements
3. Team interpretation of the project scope
4. Proposed minimum working end-to-end baseline
5. Proposed system architecture
6. Initial modules and their responsibilities
7. Proposed interfaces and shared data structures
8. Separation between the simulator-independent core and simulator adapters
9. Development phases and milestones
10. Testing and evaluation strategy
11. Documentation and reproducibility strategy
12. Team workflow and pull-request process
13. Responsible AI-use and review expectations
14. Known risks and dependencies
15. Agreed decisions
16. Provisional decisions
17. Unresolved questions
18. Immediate next actions

Label important statements as one of:

- ASSIGNMENT REQUIREMENT
- TEAM DECISION
- PROVISIONAL DECISION
- AI RECOMMENDATION
- UNRESOLVED

Do not invent team agreement.
Do not silently resolve disagreements.
Do not claim that code, tests, metrics or results already exist.
Do not generate implementation code.

Show us the complete draft before writing the file.
```

---

# Part 5: Review and Agree on the Master Plan

The generated master plan is a draft, not an approved team decision.

Every team member should read it before it is added to the group repository.

## Review questions

### Assignment alignment

- Does the plan include all mandatory Assignment 1 capabilities and deliverables?
- Has an AI recommendation been incorrectly presented as an assignment requirement?
- Has the plan added unnecessary requirements?

### Scope

- Is the proposed minimum baseline achievable?
- Is the plan too ambitious for the available time?
- Does it distinguish the first working baseline from later extensions?
- Are important dependencies visible?

### Architecture

- Is the core library independent of the simulator?
- Are module responsibilities clear?
- Are interfaces explicit enough to allow parallel work?
- Has the plan committed prematurely to an algorithm or data structure?

### Evidence

- Does the plan explain how each major capability will be tested?
- Are evaluation claims measurable or observable?
- Does it include failure cases and limitations?
- Can a new user reproduce the demonstration?

### Team agreement

- Which decisions have genuinely been agreed upon?
- Which decisions remain provisional?
- Which decisions require a small experiment?
- Has the AI hidden or resolved a disagreement?
- Can every team member explain the plan?

## Ask AI to critique its draft

```text
Critique the current project master plan as a senior engineering reviewer.

Identify:

1. missing Assignment 1 requirements;
2. areas where the scope is too broad or too vague;
3. unsupported assumptions;
4. hidden simulator coupling;
5. unclear module boundaries;
6. untestable success criteria;
7. recommendations presented as team decisions;
8. decisions that require team agreement or experimental evidence;
9. risks that the current plan does not address.

Do not revise the file yet. Ask us to resolve the three most important issues
one at a time.
```

Discuss the critique and revise the plan as a team.

## Record agreement

At the end of the master plan, include an approval section:

```markdown
## Team Review

- Review date: YYYY-MM-DD
- Team members who participated:
  - Name
  - Name
  - Name
- Status: Agreed initial master plan
- Unresolved matters:
  - Item 1
  - Item 2
- Next review point: Week X or project milestone
```

Agreement does not mean the plan can never change. It means that the current version accurately represents the team's shared starting point.

---

# Part 6: Share Contributions Through Personal Forks

Once the team agrees on the initial master plan, divide the next actions into focused contributions.

Possible initial contributions include:

- refining the package architecture;
- investigating simulator options;
- defining simulator-independent data structures;
- creating the package boilerplate;
- preparing testing and linting configuration;
- setting up documentation;
- investigating lane and drivable-area detection;
- identifying suitable evaluation data and evidence;
- preparing the initial CI workflow.

Each contribution should be small enough to review independently.

## Step 1: Synchronise your fork

```bash
git fetch upstream
git switch main
git merge upstream/main
git push origin main
```

## Step 2: Create a focused branch

```bash
git switch -c planning/refine-module-interfaces
```

Use a branch name that clearly describes the contribution.

## Step 3: Make and review your contribution

Before committing:

- inspect all changed files;
- run relevant checks;
- confirm that the contribution follows the agreed master plan;
- ensure that you can explain every submitted change;
- record significant AI interactions;
- distinguish your decisions from AI suggestions.

## Step 4: Commit and push to your fork

```bash
git add .
git commit -m "docs: propose initial module interfaces"
git push -u origin planning/refine-module-interfaces
```

If AI materially assisted the work, declare it in the commit body. For example:

```text
docs: propose initial module interfaces

Added an initial description of the information exchanged between perception,
planning and control components.

AI-assisted: GitHub Copilot helped compare interface alternatives and draft the
initial text. I reviewed the proposal against the team master plan and revised
the final decisions before committing.
```

## Step 5: Open a pull request to the group repository

Open a pull request from your branch in your personal fork to the appropriate branch in the shared group repository.

The pull request should explain:

- the purpose of the contribution;
- the relevant part of the master plan;
- the important decisions made;
- alternatives considered;
- testing or evidence provided;
- significant AI assistance;
- unresolved concerns for reviewers.

At least one other team member should review the contribution before it is merged.

> Sharing a contribution means more than sending files. It means giving teammates a reviewable change with a clear rationale and evidence.

---

# Part 7: Suggested Pull Request Template

```markdown
## Purpose

What part of the project master plan does this contribution address?

## Changes

What files, interfaces, documentation or configuration have changed?

## Engineering decisions

What decisions were made, and why?

## Alternatives considered

What other approaches were considered or rejected?

## Evidence and verification

What checks, tests, examples or references support this contribution?

## AI assistance

How was AI used? What was accepted, modified or rejected?

## Questions for reviewers

What should reviewers examine carefully?

## Unresolved matters

What remains incomplete or uncertain?
```

---

# Part 8: AI Journal Guidance

Your AI Journal should show the development of your thinking, not only the AI's final response.

---

# Completion Checklist

Before moving from planning to substantial implementation, confirm that:

- [ ] every team member has forked the group repository;
- [ ] every fork has the group repository configured as `upstream`;
- [ ] the team has reviewed the Assignment 1 specification;
- [ ] the AI interview has been completed as a group or individually;
- [ ] individual responses have been compared where applicable;
- [ ] the team has discussed the major project decisions;
- [ ] assignment requirements are separated from AI recommendations;
- [ ] the AI has drafted `docs/project-masterplan.md`;
- [ ] every team member has reviewed the master plan;
- [ ] disagreements and unresolved questions remain visible;
- [ ] the team has agreed on the current version of the plan;
- [ ] immediate next actions have been assigned;
- [ ] contributions will be made through focused branches and pull requests;
- [ ] significant AI interactions are recorded in the AI Journal;
- [ ] team members can explain the master plan and their own contributions.

---

# Final Reminder

The master plan is a living engineering document. Your team may revise it as you obtain new evidence, encounter technical constraints or improve your understanding of the assignment.

When changing the plan, record:

1. what changed;
2. why it changed;
3. who participated in the decision;
4. what evidence motivated the change;
5. which tasks, interfaces or modules are affected.

> **AI can facilitate the planning meeting and draft the master plan. Your team must interpret the assignment, negotiate the scope, approve the decisions and verify the resulting work.**
