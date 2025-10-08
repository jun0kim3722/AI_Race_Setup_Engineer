# AI Race Setup Engineer
This project helps users analyze and optimize **Assetto Corsa Competizione (ACC)** car setups, even without prior technical knowledge or extensive data visualization skills. The Large Language Model (LLM) takes your MoTeC telemetry data and setup information, then provides recommendations based on your feedback and preferences.
<img width="2780" height="1502" alt="Screenshot from 2025-10-08 11-11-44" src="https://github.com/user-attachments/assets/156064ac-ce31-402b-b828-fbec5a9e1f4f" />



### Currently Supported Cars
- Porsche 992 GT3 R
- McLaren 720S GT3 Evo
- Ferrari 296 GT3
- Mercedes AMG GT3 Evo
- BMW M4 GT3

### Planned Features
- Support for additional cars
- Docker and web interface for easier access and usability

<br>

## Install
Clone the repository and install the required Python libraries
```
git clone https://github.com/jun0kim3722/AI_Race_Setup_Engineer
pip install -r requirements.txt
```
### Setup `.env` file
> If you do not have OpenRouter API Key, please make your personal key [here](https://openrouter.ai/)

Set your API key and LLM model:
```
# This project was tested with deepseek/deepseek-chat-v3.1:free and nvidia/nemotron-nano-9b-v2:free
echo "OPENROUTER_API_KEY=<your key here>" > .env
echo "OPENROUTER_MODEL=<your LLM model name here>" > .env
```

Add the MoTeC and ACC setup directories:
```
echo "TELEMETRY_DIR=<your MoTeC file directory>" > .env
echo "SETUP_DIR=<your MoTeC file directory>" > .env
```
> On Windows, by default you can find your MoTeC and setup files in: `FIX THIS!!!!!!!!!!!!!!!`

<br>

## Run Program
For testing purposes, you can set the `.env` file as follows:
```
# LLM setup
OPENROUTER_MODEL=deepseek/deepseek-chat-v3.1:free or nvidia/nemotron-nano-9b-v2:free
OPENROUTER_API_KEY=<Your API Key>

# ACC file directory
TELEMETRY_DIR=test_ld
SETUP_DIR=ACC_setups-main
```

Run program by
```
python llm_client.py
```

<br>

## User Instruction
### Recording MoTeC Data

Start a practice session. Then navigate to:
 `setup -> current setup -> electronics -> telemetry laps`. Set `Telemetry Laps` to the number of laps you would like to record for analysis.

**Recommendation:**
- Set this value to 99 laps for a comprehensive dataset.
- Adjust your practice session settings—such as weather, tire wear, fuel consumption, and damage—to match your intended testing conditions.

<br>

### Workflow Options
This project offers several workflow options to help you optimize your car setups based on your goals and preferences:

#### Option 1: Hotlap / Qualifying Setup (Default)
> Maximize single-lap performance and peak grip.  
Use this option if your priority is achieving the fastest possible lap times, such as during hotlaps or qualifying sessions.

#### Option 2: Race Setup Adjustment
> Prioritize consistency and endurance over peak pace.  
Ideal for race conditions where long-term tire management, stability, and overall consistency are more important than outright lap speed.

#### Option 3: Custom / Specific Adjustment
> Execute a user-requested tuning (assumes user knowledge).  
Choose this option if you have a specific setup change in mind and want the AI to incorporate your custom adjustments into the car setup.

#### Option 4: Continue Previous Work
> Resume the sequential setup plan from your last session.  
This option allows you to pick up exactly where you left off, continuing a workflow or iterative setup optimization process.

<br>

### Loading MoTeC data and Setup data
When you run `python llm_client.py`, the program will list all of your MoTeC data.
Example:
```
--------- Loading MoTeC Data --------------------------------------
[10/08/25 11:32:37] INFO     Processing request of type CallToolRequest                                                                                                                                                 server.py:664
                    INFO     Processing request of type ListToolsRequest                                                                                                                                                server.py:664
No.  Track Name           Car Name             Date         Time    
1    monza                porsche_992_gt3_r    2025.09.06   22.13.11
2    nurburgring          porsche_992_gt3_r    2025.09.06   21.43.58
3    nurburgring          porsche_992_gt3_r    2025.09.06   21.24.45
4    mount_panorama       porsche_992_gt3_r    2025.09.06   20.27.50
5    mount_panorama       porsche_992_gt3_r    2025.09.06   20.27.42
6    mount_panorama       porsche_992_gt3_r    2025.09.06   20.03.41

-------------------------------------------------------------------
Enter the session number you would like to analyze: <int>

--------- Loading Setup Data --------------------------------------
[10/08/25 11:32:38] INFO     Processing request of type CallToolRequest                                                                                                                                                 server.py:664
No.  Setup Name          
1    1.3.json
2    20min.json
3    1.json
4    1.7.json
5    2stop.json
6    1.4.json
7    1.5.json
8    1.6.json
9    1.2.json

-------------------------------------------------------------------
Enter the setup number you would like to analyze: <int>
```
Please enter an integer to select the correct MoTeC data and Setup.

<br>

### Step by Step Guide
Follow these steps to analyze and optimize your car setups using the AI setup engineer:

1. **Load Initial Session and Setup Data**  
   Start by loading the session and setup data you want to analyze. This ensures the AI has the correct context for your car, track, and driving conditions.

2. **Choose a Workflow**  
   Select the workflow that best suits your purpose, such as hotlap optimization, race setup adjustment, custom tuning, or continuing a previous session. Each workflow tailors the AI’s recommendations to your specific goals.

3. **Provide Feedback on Car Behavior**  
   Give the AI setup engineer detailed feedback about how the car feels on track.  
   *Examples:*  
   - "The car still oversteers on exit."  
   - "Turn 3 entry understeer is too strong."  
   Accurate feedback helps the AI suggest meaningful setup adjustments.

4. **Test Suggested Setup Changes**  
   After the AI proposes changes (e.g., "Front ARB +1"), apply the new setup and test it on track. This allows you to evaluate the impact of the adjustments in real driving conditions.

5. **Load New Test Data**  
   Type `read` or `r` to load the latest telemetry and setup data. This keeps the AI informed about your most recent session and ensures recommendations are based on current performance.

6. **Provide Updated Feedback**  
   Based on the new test results, provide updated feedback to the AI. This iterative process refines the car setup progressively until it meets your goals.

7. **Continue Communicating with the AI Engineer**  
   Repeat steps 4–6 as necessary, continuing the iterative loop until your setup is optimized.

8. **End the Session**  
   When finished, type `exit`/`e` or `quit`/`q` to end the session.  
   You can always save the current chat and continue later, maintaining continuity between sessions.

<br>

## TODO List
- [ ] Add setup guide for LLM
- [ ] Dockerize
- [ ] Connect to web
- [ ] Add multiple language support

<br>

## Author:
***Junyoung Kim @ Purdue University ECE***
- Email: kim3722@purdue.edu
- LinkedIn: [Junyoung (Jun) Kim](https://www.linkedin.com/in/jun0kim0329/)
- Portfolio: [Jun Kim's web](https://jun0kim.vercel.app/)

<br>

## Reference
This project incorporates code originally written by [gotzl](https://github.com/gotzl/ldparser) for MoTeC data parser.

ACC setup information was referenced from the [Coach Dave Academy](https://coachdaveacademy.com/tutorials/the-ultimate-acc-car-setup-guide/).

