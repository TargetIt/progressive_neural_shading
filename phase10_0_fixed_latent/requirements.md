# Phase 1.0 Requirements

## New Features (2026-05-16)

- **Minimal App Framework** (`app.py`): window + GPU device + blit, ~100 lines
- **Minimal Blit Shader** (`app.slang`): Tensor → screen output texture
- **Hello Slang Shader** (`step_1_0_hello.slang`): returns solid red `float3(1,0,0)`
- **Entry Point** (`step_1_0_hello.py`): loads shader, render loop, ESC to exit

## Functional Requirements

1. User can run `python src/step_1_0_hello.py` and see a red window
2. Window size is 512×512
3. ESC key closes the window
4. All pixels should be (1.0, 0.0, 0.0)

## Acceptance Criteria

- [x] Window opens without errors
- [x] All pixels are solid red
- [x] ESC closes the window
- [x] shader compiles successfully (first run may take 2-5 seconds)
