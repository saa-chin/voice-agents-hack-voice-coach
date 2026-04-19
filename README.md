<img src="assets/banner.png" alt="Logo" style="border-radius: 30px; width: 60%;">

## Documentation

| # | Document |
| --- | --- |
| 1 | [The Problem](./docs/01-the-problem.md) |
| 2 | [The Solution](./docs/02-the-solution.md) |
| 3 | [The Architecture](./docs/03-the-architecture.md) |
| 4 | [Implementation Details](./docs/04-implementation-details.md) |

## Context
- Cactus (YC S25) is a low-latency engine for mobile devices & wearables. 
- Cactus runs locally on edge devices with hybrid routing of complex tasks to cloud models like Gemini.
- Google DeepMind just released Gemma 4, the first on-device model you can voice-prompt. 
- Gemma 4 on Cactus is multimodal, supporting voice, vision, function calling, transcription and more! 

## Challenge
- All teams MUST build products that use Gemma 4 on Cactus. 
- All products MUST leverage voice functionality in some way. 
- All submissions MUST be working MVPs capable of venture backing. 
- Winner takes all: Guaranteed YC Interview + GCP Credits. 

## Special Tracks 
- Best On-Device Enterprise Agent (B2B): Highest commercial viability for offline tools.
- Ultimate Consumer Voice Experience (B2C): Best use of low-latency compute to create ultra-natural, instantaneous voice interaction.
- Deepest Technical Integration: Pushing the boundaries of the hardware/software stack (e.g., novel routing, multi-agent on-device setups, extreme power optimization).

Prizes per special track: 
- 1st Place: $2,000 in GCP credits
- 2nd Place: $1,000 in GCP credits 
- 3rd Place: $500 in GCP credits 

## Judging 
- **Rubric 1**: The relevnance and realness of the problem and appeal to enterprises and VCs. 
- **Rubric 2**: Correcness & quality of the MVP and demo. 

## Setup (clone this repo and hollistically follow)
- Step 1: Fork this repo, clone to your Mac, open terminal.
- Step 2: `git clone https://github.com/cactus-compute/cactus`
- Step 3: `cd cactus && source ./setup && cd ..` (re-run in new terminal)
- Step 4: `cactus build --python`
- Step 5: `cactus download google/functiongemma-270m-it --reconvert`
- Step 6: Get cactus key from the [cactus website](https://cactuscompute.com/dashboard/api-keys)
- Sept 7: Run `cactus auth` and enter your token when prompted.
- Step 8: `pip install google-genai` (if using cloud fallback) 
- Step 9: Obtain Gemini API key from [Google AI Studio](https://aistudio.google.com/api-keys) (if using cloud fallback) 
- Step 10: `export GEMINI_API_KEY="your-key"` (if using cloud fallback) 

## Run on a phone
Two one-shot scripts at the repo root build the React Native app and launch it on a connected device. Both handle tool checks, dep install, device detection, and the build, and fail with clear, actionable messages.

### iPhone

```sh
./run-ios
```

Flags:
- `./run-ios --list` — show connected (and known-offline) iOS devices
- `./run-ios --device "iPhone"` — pick a device by name or UDID when more than one is plugged in
- `./run-ios --release` — build the Release configuration
- `./run-ios --clean` — wipe `build/`, `Pods/`, `node_modules/`, and DerivedData first
- `./run-ios --skip-pods` / `--skip-install` — skip pod or npm install
- `./run-ios --help` — full usage

First-time iPhone checklist (the script will remind you of these on failure):
1. Plug the iPhone into the Mac with a data cable, unlock it, tap "Trust This Computer".
2. Enable Developer Mode: Settings → Privacy & Security → Developer Mode.
3. Open `mobile/ios/AIVoiceCoach.xcworkspace` in Xcode once and pick a Team under Signing & Capabilities.
4. After the first install, on the iPhone trust the developer cert: Settings → General → VPN & Device Management.

### Android

```sh
./run-android
```

Flags:
- `./run-android --list` — show online and unauthorized/offline Android devices
- `./run-android --device "<adb serial>"` — pick a device when more than one is connected
- `./run-android --release` — build the Release variant
- `./run-android --clean` — `./gradlew clean` and remove `node_modules/`
- `./run-android --skip-install` — skip npm install
- `./run-android --help` — full usage

First-time Android checklist (the script will remind you of these on failure):
1. Install Android Studio once so the SDK + platform-tools are present (the script auto-detects `~/Library/Android/sdk` on macOS or `~/Android/Sdk` on Linux).
2. Use JDK 17 (RN 0.85): `export JAVA_HOME=$(/usr/libexec/java_home -v 17)`.
3. On the phone: tap Build Number 7× to unlock Developer Options, then enable USB Debugging.
4. Plug in over a data cable, unlock the phone, and tap "Allow" on the RSA fingerprint prompt.
5. The script auto-runs `adb reverse tcp:8081 tcp:8081` so Metro reaches the device over USB.

## Next steps
1. Read Cactus docs carefully: [Link](https://docs.cactuscompute.com/latest/)
2. Read Gemma 4 on Cactus walkthrough carefully: [Link](https://docs.cactuscompute.com/latest/blog/gemma4/)
3. Cactus & DeepMind team would be available on-site. 