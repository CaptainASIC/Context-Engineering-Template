# 🚀 Context Engineering & Claude Code Template

A comprehensive template for getting started with Context Engineering and Claude Code - the discipline of engineering context for AI coding assistants so they have the information necessary to get the job done end to end.

> **Context Engineering is 10x better than prompt engineering and 100x better than vibe coding.**

## 📋 Prerequisites

- Terminal/Command line access
- Node.js installed (for Claude Code installation)
- GitHub account (for GitHub CLI integration)
- Text editor (VS Code recommended)

## 🔧 Installation

**macOS/Linux:**
```bash
npm install -g @anthropic-ai/claude-code
```

**Windows (WSL recommended):**
See detailed instructions in [install_claude_code_windows.md](./install_claude_code_windows.md)

**Verify installation:**
```bash
claude --version
```

## 🚀 Quick Start

```bash
# 1. Clone this template
git clone https://github.com/CaptainASIC/Context-Engineering-Template.git
cd Context-Engineering-Template

# 2. Set up your project rules (optional - template provided)
# Edit CLAUDE.md to add your project-specific guidelines

# 3. Add examples (highly recommended)
# Place relevant code examples in the examples/ folder

# 4. Create your initial feature request
# Edit INITIAL.md with your feature requirements

# 5. Create your project planning documentation
# Edit PLANNING.md with your project plans

# 6. Creat your project tasks documentation
# Edit TASKS.md with your project tasks

# 7. Generate a comprehensive PRP (Product Requirements Prompt)
# In Claude Code, run:
/generate-prp INITIAL.md

# 8. Execute the PRP to implement your feature
# In Claude Code, run:
/execute-prp PRPs/your-feature-name.md
```

## 📚 Table of Contents

- [What is Context Engineering?](#what-is-context-engineering)
- [Template Structure](#template-structure)
- [Step-by-Step Workflow](#step-by-step-workflow)
- [Claude Code Setup](#claude-code-setup)
- [Context Engineering with Examples](#context-engineering-with-examples)
- [The PRP Workflow](#the-prp-workflow)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering represents a paradigm shift from traditional prompt engineering:

### Prompt Engineering vs Context Engineering

**Prompt Engineering:**
- Focuses on clever wording and specific phrasing
- Limited to how you phrase a task
- Like giving someone a sticky note

**Context Engineering:**
- A complete system for providing comprehensive context
- Includes documentation, examples, rules, patterns, and validation
- Like writing a full screenplay with all the details

### Why Context Engineering Matters

1. **Reduces AI Failures**: Most agent failures aren't model failures - they're context failures
2. **Ensures Consistency**: AI follows your project patterns and conventions
3. **Enables Complex Features**: AI can handle multi-step implementations with proper context
4. **Self-Correcting**: Validation loops allow AI to fix its own mistakes

## Template Structure

```
context-engineering-template/
├── .claude/
│   ├── commands/
│   │   ├── generate-prp.md    # Generates comprehensive PRPs
│   │   └── execute-prp.md     # Executes PRPs to implement features
│   ├── agents/                # Subagents for specialized tasks
│   ├── hooks/                 # Automation hooks
│   └── settings.local.json    # Claude Code permissions
├── PRPs/
│   ├── templates/
│   │   └── prp_base.md       # Base template for PRPs
│   └── EXAMPLE_multi_agent_prp.md  # Example of a complete PRP
├── examples/                  # Your code examples (critical!)
├── CLAUDE.md                 # Global rules for AI assistant
├── PLANNING.md               # Project architecture and goals
├── TASK.md                   # Task tracking and management
├── INITIAL.md               # Template for feature requests
├── INITIAL_EXAMPLE.md       # Example feature request
└── README.md                # This file
```

## Step-by-Step Workflow

The recommended workflow follows this sequence: **Initial Context Building → Planning & Task Files → PRP Generation → Codebase Building**

### 1. Initial Context Building (CLAUDE.md)

Set up context files that Claude automatically pulls into every conversation, containing project-specific information, commands, and guidelines.

```bash
mkdir your-project-name && cd your-project-name
claude
```

Use the built-in command:
```
/init
```

Or create your own CLAUDE.md file based on the template in this repository. The `CLAUDE.md` file contains project-wide rules that the AI assistant will follow in every conversation:

- **Project awareness**: Reading planning docs, checking tasks
- **Code structure**: File size limits, module organization
- **Testing requirements**: Unit test patterns, coverage expectations
- **Style conventions**: Language preferences, formatting rules
- **Documentation standards**: Docstring formats, commenting practices

### 2. Planning and Task Files

Create your project structure and planning documents:

**PLANNING.md**: Define your project's architecture, goals, style, and constraints. This serves as the single source of truth for the project's direction.

**TASK.md**: Track ongoing and pending work. Before starting any new development task, check this file. If the task isn't listed, add it with a brief description and today's date.

### 3. PRP Generation

Create your initial feature request in `INITIAL.md`:

```markdown
## FEATURE:
[Describe what you want to build - be specific about functionality and requirements]

## EXAMPLES:
[List any example files in the examples/ folder and explain how they should be used]

## DOCUMENTATION:
[Include links to relevant documentation, APIs, or MCP server resources]

## OTHER CONSIDERATIONS:
[Mention any gotchas, specific requirements, or things AI assistants commonly miss]
```

Then generate a comprehensive PRP:
```bash
/generate-prp INITIAL.md
```

### 4. Codebase Building

Execute the PRP to implement your feature:
```bash
/execute-prp PRPs/your-feature-name.md
```

## Claude Code Setup

### Permission Management

Configure tool allowlists to streamline development while maintaining security:

**Method 1: Interactive Allowlist**
When Claude asks for permission, select "Always allow" for common operations.

**Method 2: Use /permissions command**
```
/permissions
```

**Method 3: Create project settings file**
Create `.claude/settings.local.json`:
```json
{
  "allowedTools": [
    "Edit",
    "Read",
    "Write",
    "Bash(git add:*)",
    "Bash(git commit:*)",
    "Bash(npm:*)",
    "Bash(python:*)",
    "Bash(pytest:*)"
  ]
}
```

### Custom Slash Commands

Slash commands are the key to adding your own workflows into Claude Code. They live in `.claude/commands/` and enable you to create reusable, parameterized workflows.

**Built-in Commands:**
- `/init` - Generate initial CLAUDE.md
- `/permissions` - Manage tool permissions
- `/clear` - Clear context between tasks
- `/agents` - Manage subagents
- `/help` - Get help with Claude Code

**Custom Commands Example:**
```markdown
# Command: analyze-performance

Analyze the performance of the file specified in $ARGUMENTS.

## Steps:
1. Read the file at path: $ARGUMENTS
2. Identify performance bottlenecks
3. Suggest optimizations
4. Create a benchmark script
```

### MCP Server Integration

Connect Claude Code to Model Context Protocol (MCP) servers for enhanced functionality:

```bash
# Install Serena for semantic code analysis and editing
claude mcp add serena -- uvx --from git+https://github.com/oraios/serena serena start-mcp-server --context ide-assistant --project $(pwd)
```

## Context Engineering with Examples

Transform your development workflow from simple prompting to comprehensive context engineering.

### The PRP Framework

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints that include:

- Complete context and documentation
- Implementation steps with validation
- Error handling patterns
- Test requirements

### Defining Your Requirements

Your INITIAL.md should always include:

```markdown
## FEATURE
Build a user authentication system

## EXAMPLES
- Authentication flow: `examples/auth-flow.js`
- Similar API endpoint: `src/api/users.js` 
- Database schema pattern: `src/models/base-model.js`
- Validation approach: `src/validators/user-validator.js`

## DOCUMENTATION
- JWT library docs: https://github.com/auth0/node-jsonwebtoken
- Our API standards: `docs/api-guidelines.md`

## OTHER CONSIDERATIONS
- Use existing error handling patterns
- Follow our standard response format
- Include rate limiting
```

### Critical PRP Strategies

**Examples**: The most powerful tool - provide code snippets, similar features, and patterns to follow

**Validation Gates**: Ensure comprehensive testing and iteration until all tests pass

**No Vibe Coding**: Validate PRPs before executing them and the code after execution!

## The PRP Workflow

### How /generate-prp Works

1. **Research Phase**
   - Analyzes your codebase for patterns
   - Searches for similar implementations
   - Identifies conventions to follow

2. **Documentation Gathering**
   - Fetches relevant API docs
   - Includes library documentation
   - Adds gotchas and quirks

3. **Blueprint Creation**
   - Creates step-by-step implementation plan
   - Includes validation gates
   - Adds test requirements

4. **Quality Check**
   - Scores confidence level (1-10)
   - Ensures all context is included

### How /execute-prp Works

1. **Load Context**: Reads the entire PRP
2. **Plan**: Creates detailed task list
3. **Execute**: Implements each component
4. **Validate**: Runs tests and linting
5. **Iterate**: Fixes any issues found
6. **Complete**: Ensures all requirements met

## Advanced Features

### Subagents for Specialized Tasks

Subagents are specialized AI assistants that operate in separate context windows with focused expertise:

```markdown
---
name: security-auditor
description: "Security specialist. Proactively reviews code for vulnerabilities and suggests improvements."
tools: Read, Grep, Glob
---

You are a security auditing specialist focused on identifying and preventing security vulnerabilities...
```

### Automation with Hooks

Hooks provide deterministic control over Claude Code's behavior through user-defined shell commands:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/log-tool-usage.sh"
          }
        ]
      }
    ]
  }
}
```

### GitHub CLI Integration

Set up the GitHub CLI to enable Claude to interact with GitHub:

```bash
# Install and authenticate GitHub CLI
gh auth login

# Use custom commands
/fix-github-issue 123
```

### Parallel Development with Git Worktrees

Use Git worktrees to enable multiple Claude instances working on independent tasks:

```bash
# Create worktrees for different features
git worktree add ../project-auth feature/auth
git worktree add ../project-api feature/api

# Launch Claude in each worktree
cd ../project-auth && claude  # Terminal 1
cd ../project-api && claude   # Terminal 2
```

## Using Examples Effectively

The `examples/` folder is **critical** for success. AI coding assistants perform much better when they can see patterns to follow.

### What to Include in Examples

1. **Code Structure Patterns**
   - How you organize modules
   - Import conventions
   - Class/function patterns

2. **Testing Patterns**
   - Test file structure
   - Mocking approaches
   - Assertion styles

3. **Integration Patterns**
   - API client implementations
   - Database connections
   - Authentication flows

4. **CLI Patterns**
   - Argument parsing
   - Output formatting
   - Error handling

### Example Structure

```
examples/
├── README.md           # Explains what each example demonstrates
├── cli.py             # CLI implementation pattern
├── agent/             # Agent architecture patterns
│   ├── agent.py      # Agent creation pattern
│   ├── tools.py      # Tool implementation pattern
│   └── providers.py  # Multi-provider pattern
└── tests/            # Testing patterns
    ├── test_agent.py # Unit test patterns
    └── conftest.py   # Pytest configuration
```

## Best Practices

### 1. Be Explicit in INITIAL.md
- Don't assume the AI knows your preferences
- Include specific requirements and constraints
- Reference examples liberally

### 2. Provide Comprehensive Examples
- More examples = better implementations
- Show both what to do AND what not to do
- Include error handling patterns

### 3. Use Validation Gates
- PRPs include test commands that must pass
- AI will iterate until all validations succeed
- This ensures working code on first try

### 4. Leverage Documentation
- Include official API docs
- Add MCP server resources
- Reference specific documentation sections

### 5. Customize CLAUDE.md
- Add your conventions
- Include project-specific rules
- Define coding standards

### 6. Follow the Workflow
- Start with context building (CLAUDE.md)
- Create planning and task files (PLANNING.md, TASK.md)
- Generate PRPs for features
- Build codebase systematically

### 7. Advanced Prompting Techniques

**Power Keywords**: Claude responds to certain keywords with enhanced behavior:
- **IMPORTANT**: Emphasizes critical instructions
- **Proactively**: Encourages Claude to take initiative
- **Ultra-think**: Can trigger more thorough analysis (use sparingly)

**Essential Tips**:
- Avoid prompting for "production-ready" code - this often leads to over-engineering
- Prompt Claude to write scripts to check its work
- Avoid backward compatibility unless specifically needed
- Focus on clarity and specific requirements

## Resources

- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [MCP Documentation](https://docs.anthropic.com/en/docs/claude-code/mcp)
- [Subagents Documentation](https://docs.anthropic.com/en/docs/claude-code/sub-agents)
- [Hooks Documentation](https://docs.anthropic.com/en/docs/claude-code/hooks)
- [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)

---
