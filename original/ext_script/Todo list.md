## What I’d do next, in order (fastest progress)

### 1. Build one iiwa cross-DOF model with mask + masked loss (prove “one model, many DOFs”).

### 2. Implement context encoder + FK predictor (still on iiwa first; pretend 5/6/7 are “different tasks”).

### 3. Add robot randomization in sim and retrain the context model across many robots.

### 4. Deploy: collect a few hundred real samples → infer z → FK black box + IK by optimization.