# Cuthrough Development Guide

## Environment Setup

### Prerequisites
- Install Miniconda or Anaconda
- Git installed on your machine
- Access to the Cuthrough repository
- GPG key for signed commits

### Setting Up Your Development Environment

1. Clone the repository:
```bash
git clone https://github.com/admarble/video-ai-platform.git
cd video-ai-platform
```

2. Create and activate the Conda environment:
```bash
conda create -n cuthrough_38 python=3.8
conda activate cuthrough_38
```

3. Install project dependencies:
```bash
conda install pip
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Git Workflow Best Practices

### Before Starting Work

1. Always start with the latest code:
```bash
git pull origin main
```

2. Create a new branch for your feature/fix:
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### During Development

1. Regularly check status of your changes:
```bash
git status
```

2. Stage and commit changes frequently with signed commits:
```bash
git add <changed-files>
git commit -S -m "type: brief description of changes"
```

Commit message types:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks
- `deps:` for dependency updates
- `security:` for security improvements

### Before Pushing Changes

1. Review your changes:
```bash
git diff origin/main
```

2. Check commit history:
```bash
git log --oneline
```

3. Update your branch with main if needed:
```bash
git checkout main
git pull origin main
git checkout your-branch
git merge main
```

4. Push your changes:
```bash
git push origin your-branch
```

### Code Review Process

1. Go to GitHub and create a Pull Request (PR)
2. Fill in the PR template with:
   - Description of changes
   - Related issues
   - Testing performed
   - Screenshots (if UI changes)
3. Ensure all required checks pass:
   - Tests (Python 3.8)
   - Linting
   - Security scan
   - Documentation build
4. Get approval from code owners
5. Resolve all conversations
6. Ensure branch is up to date
7. Merge only when all requirements are met

### Branch Protection Rules

The `main` branch is protected with the following rules:
1. Pull Request Requirements:
   - At least one approval required
   - Code owner review required
   - Stale reviews are dismissed
   - Latest changes must be approved
   - All conversations must be resolved

2. Status Checks:
   - All tests must pass
   - Linting must pass
   - Security scan must pass
   - Documentation must build
   - Branch must be up to date

3. Commit Requirements:
   - All commits must be signed
   - Linear history required (no merge commits)
   - No force pushes allowed
   - No branch deletions

### Automated Security Updates

Dependabot is configured to:
1. Check for updates weekly (Mondays at 09:00 UTC)
2. Create PRs for:
   - Security updates (high priority)
   - Dependency updates (grouped by type)
   - GitHub Actions updates
3. Apply security patches automatically after approval
4. Group updates to reduce PR noise:
   - Security updates separately
   - Minor and patch updates together
   - Major updates individually

### Best Practices

1. **Keep commits atomic**: Each commit should represent one logical change

2. **Write descriptive commit messages**:
```
feat: add video compression feature

- Implement adaptive compression algorithm
- Add quality settings configuration
- Update documentation for compression options
```

3. **Regular updates**:
   - Pull from main daily
   - Push your changes at least once per day
   - Keep PRs small and focused

4. **Code Organization**:
   - Follow the project structure
   - Keep files focused and single-purpose
   - Use meaningful file and variable names

5. **Documentation**:
   - Update README.md when adding features
   - Add comments for complex logic
   - Include docstrings for functions

6. **Testing**:
   - Write tests for new features
   - Run existing tests before committing
   - Update tests when changing functionality

### Common Issues and Solutions

1. **Merge Conflicts**:
```bash
git status  # Check which files are conflicting
# Resolve conflicts in your editor
git add <resolved-files>
git commit -S -m "fix: resolve merge conflicts"
```

2. **Undo Last Commit** (not pushed):
```bash
git reset --soft HEAD~1
```

3. **Discard Local Changes** (careful!):
```bash
git checkout -- <file>  # For specific file
git reset --hard HEAD  # For all changes
```

## Project Structure

```
cuthrough/
├── src/                 # Source code
├── tests/              # Test files
├── requirements.txt    # Dependencies
├── .gitignore         # Git ignore rules
└── README.md          # Project documentation
```

## Getting Help

1. Check the project documentation
2. Review existing issues on GitHub
3. Ask team members in the development channel
4. Create a new issue for bugs or feature requests

Remember: When in doubt, ask! It's better to ask questions than to make assumptions. 