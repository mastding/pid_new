# Architecture Notes

This directory stores architecture planning documents for the next-stage evolution of `pid_v2`.

Documents:

- `skill_provider_refactor_plan.md`
  Architecture refactor plan for moving from direct algorithm calls to a `pipeline + skill + provider + policy` model.
- `first_batch_skills_design.md`
  Detailed design for the first batch of five high-value skills to implement in the current project.

Design goals:

- Keep the tuning pipeline stable while making algorithms replaceable.
- Separate business capability orchestration from concrete algorithm implementations.
- Make future window detection, dead-time estimation, identification, tuning, and evaluation logic easy to swap and test.
