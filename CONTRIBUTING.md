# Contributing and adding projects

Create each new project under `projects/<project-name>/` and use `PROJECT_TEMPLATE.md` as the documentation checklist.

Before adding a project:

1. Confirm the dataset's source, license, and sharing permissions.
2. Keep notebooks reproducible from top to bottom.
3. Move reusable training and inference logic into `src/`.
4. Compare against a simple baseline.
5. Add appropriate tests and record exact metrics.
6. Include at least one useful visual with readable labels.
7. Document limitations and unsafe or unsupported uses.
8. Add the project to the root README only after it is runnable.

Run the relevant project's test command before committing changes.
