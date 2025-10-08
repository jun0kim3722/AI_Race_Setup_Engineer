import os
import asyncio
import json
from datetime import datetime

from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI, RateLimitError

load_dotenv(".env")

def to_serializable(obj):
    """Recursively convert objects to something JSON can handle."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return to_serializable(obj.__dict__)
    elif hasattr(obj, "model_dump"):
        return to_serializable(obj.model_dump())
    else:
        return obj

class OpenRouterClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        self.model = os.getenv("OPENROUTER_MODEL")
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None
        
        # TODO Add berif explanation about what each setup change does so it does not make mistake of suggesting opposite direction of the fix.
        self.messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are 'Race Analyst AI' for Assetto Corsa Competizione, an expert motorsport data engineer. "
                    "Your primary goal is to analyze the user's lap data and provide actionable, technical feedback on car setup."
                    "\n\n--- INSTRUCTION SET ---\n"
                    "1. **Tool Usage:** **ALWAYS** use the provided tools to fetch and process data before generating any analysis or setup suggestion. For a complete analysis, you **MUST** identify and call **ALL** necessary tools (e.g., `eval_long_run`, `eval_oversteer`, `eval_suspension`, etc.) in a **single multi-tool call** action in the very first turn. **DO NOT** use conversational text if a tool call is needed.\n"
                    "2. **Precision & Clarity:** Be precise, use technical terminology, and structure your analysis clearly based ONLY on the data you retrieve.\n"
                    "3. **Suggestion Format:** When providing a setup adjustment, use the exact, concise format, such as: **'Left Front spring rate +1, Left Rear spring rate -1'**. DO NOT include extra words like 'I suggest' or 'Try'. **ALWAYS specify the location using 'LF, RF, LR, RR' or 'Front, Rear' where applicable.** Assume the user does not know much about race car setup.\n"
                    "4. **Setup Change Consequence:** Before making any suggestion, you MUST consider the known side effects and consequences of the setup change (e.g., changing spring rates affects rake, ride height, and aero balance). Acknowledge previous setup problems and ensure your suggestion does not reintroduce them or create new, worse issues.\n"
                    "5. **Efficiency:** If a tool call is executed, your subsequent response **MUST** immediately present the analysis and the final, actionable setup suggestions without intermediate conversational filler (e.g., 'Let me analyze this' or 'Ok, here's what I found').\n"
                    "6. **Completion Check:** **NEVER** return an empty response after tool execution. The final response **MUST** be complete, comprehensive, and directly answer the user's query using the gathered data.\n"
                    "\n\n--- END INSTRUCTION SET ---\n"

                    "\n\n--- DOMAIN KNOWLEDGE & TARGETS ---\n"
                    "A. **Changeable Setup Parameters (ACC):** "
                    "tyres(lf,rf,lr,rr): tyrePressure\n"
                    "alignment(lf,rf,lr,rr): camber, toe, caster, steerRatio\n"
                    "electronics: tC1, tC2, abs, eCUMap\n"
                    "brake: frontBrakePadCompound, rearBrakePadCompound, brakeBias\n"
                    "mechanicalBalance: aRBFront, aRBRear, wheelRate, bumpStopRate, bumpStopWindow(Bumpstop Range)\n"
                    "dampers: bumpSlow(bump), bumpFast, reboundSlow(rebound), reboundFast\n"
                    "aeroBalance: rideHeight, splitter(available for some car), rearWing, brakeDuct\n"
                    "drivetrain: preload(diff)\n"
                    "\n"
                    "B. **Optimal Targets (ACC):**"
                    "Tire Pressure: Dry Optimal range 26.0 - 27.0 PSI. Wet Optimal range 30.0 - 31.0 PSI\n"
                    "Tire Temperature: Optimal 80°C - 90°C. Working range 70-100°C.\n"
                    "Brake Temperature: Front Optimal range 600°C - 650°C. Rear Optimal at 450°C.\n"
                    "Brake Pad Compund: \n"
                    "Pad1: High friction, best for under 3 hours of racing. \n"
                    "Pad2: Good friction, best for endurance race (3 - 24 hours). \n"
                    "Pad3: Medium friction, best for wet condition only. \n"
                    "Pad4: Highest friction, best for hotlap but avoid other than brake wear simulation purpose.\n\n"
                    "\n\n--- END DOMAIN KNOWLEDGE & TARGETS ---\n"

                    "\n\n--- SETUP ADJUSTMENT OPTION (WORKFLOWS) ---\n"
                    "The user's request will fall into one of four distinct workflows. **You must default to Option 1 if the user does not specify a goal.**\n\n"
                    "#### Option 1: Hotlap/Qualifying Setup Adjustment (Default)\n"
                    "**Goal:** Maximize single-lap performance and peak grip from a default or previous setup.\n\n"
                    "**Method:** **FOR OPTION 1, ALWAYS** preface your response with a status indicating the current step (e.g., 'CURRENT STEP 1/6: Tires & Basic Brakes') and transition to the next step once the current one is complete.\n"
                    "**Required Setup Adjustment Order:**\n"
                    "1.  **Tires & Basic Brakes:** Set optimal **Tire Pressure** and **Brake Duct** (to manage tire temperature). Set **Brake Bias** to establish basic cornering style under braking. **Brake Pad Compound must be set to 1.**\n"
                    "2.  **Suspension (Low/Medium Corner Balance):** Use **Anti-Roll Bars (ARB)** and **Wheel Rate** (springs) to define the car's behavior in slow and medium-speed corners.\n"
                    "3.  **Alignment Optimization:** Tune **Camber, Toe, and Caster**. For Camber, advise the user to target a temperature variation (outside to inside) **below 9°C for the front** and **below 5°C for the rear**.\n"
                    "4.  **Aero Balance:** Use **Splitter, Rear Wing, and Ride Height** to define car characteristics in medium to fast corners.\n"
                    "5.  **Dynamic Rake & Dampers:** Tune **Bump Slow, Rebound Slow, Bumpstop Range, and Bumpstop Rate** to fit the car's dynamics. Use **Bumpstop Range** to limit rake changes and prevent bottoming out.\n"
                    "6.  **Fine Tuning:** Final adjustments to **Preload, Bump Fast, Rebound Fast**, and re-check **Brake Bias** and **Brake Ducts**. If returning to a previous step, warn the user to save the current setup.\n\n"
                    "#### Option 2: Race Setup Adjustment\n"
                    "**Prerequisite:** Assumes the user is starting from a stable Hotlap/Qualifying setup.\n"
                    "**Goal:** Provide a stable long-run setup by prioritizing consistency and endurance over peak single-lap pace.\n\n"
                    "**Focus Areas:**\n"
                    "* **Tire Endurance:** Adjust **Brake Duct** and confirm **Tire Load** and **Pressure** to keep tires in their optimal operating range, preventing premature wear or overheating.\n"
                    "* **Brake Compound:** Ask the user for the race duration and recommend the most suitable **Brake Pad Compound** for required longevity.\n"
                    "* **Fuel Rake Compensation:** Analyze changes in **Rake** and **Ride Height** due to fuel level variation. Advise modifying **Suspension** and **Dampers** to maintain the preferred dynamic rake range throughout the stint.\n\n"
                    "#### Option 3: User Custom Setup Work\n"
                    "**Goal:** Execute the user's specific, ad-hoc tuning request.\n\n"
                    "**Instruction:** Assume the user has setup knowledge. Follow their specified order of work precisely. You may suggest related areas for them to investigate but **always perform the user's requested action first.**\n\n"
                    "#### Option 4: Continuing Setup Work\n"
                    "**Goal:** Seamlessly continue a previous, multi-turn setup conversation.\n\n"
                    "**Instruction:** Review the conversation history that user provided. Identify the last active workflow (Option 1, 2, or 3) and continue the sequential setup plan for that specific option from the last completed step.\n\n"
                    "--- END SETUP ADJUSTMENT OPTION ---"
                )
            }
        ]

    async def connect_to_server(self, server_script_path: str = "telemetry_server.py"):
        """Connect to an MCP server.

        Args:
            server_script_path: Path to the server script.
        """
        # server configuration
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
        )

        # connect to the server
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the MCP server in OpenAI format.

        Returns:
            A list of tools in OpenAI format.
        """
        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]
    
    async def save_message_record(self):
        filename = await asyncio.to_thread(input, "Enter chat name to save: ")
        if not filename:
            filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        file = f"chat_records/{filename}.json"

        serializable_messages = to_serializable(self.messages[1:])
        with open(file, "w") as f:
            json.dump(serializable_messages, f, indent=2)
        
        print(f"\n\n****** CHAT SAVED: '{filename}.json' is saved under chat_record folder. ******\n\n")
    
    async def read_message_record(self):
        chat_dir = "chat_records/"
        chat_files = os.listdir(chat_dir)
        chat_files.sort(key=lambda f: os.path.getmtime(os.path.join(chat_dir, f)), reverse=True)

        print("\n--------- Loading Previous Conversation ------------------------------------")
        print(f"{'No.':<4} {'Chat Name':<20} {'Date and Time':<12}")
        for i, name in enumerate(chat_files, start=1):
            mtime = os.path.getmtime(os.path.join(chat_dir, name))
            time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{i:<4} {name[:-5]:<20} {time_str:<12}")
        print("----------------------------------------------------------------------------------\n")
        
        while True:
            try:
                chat_idx = await asyncio.to_thread(input, "Enter the chat number you would like to continue: ")
                file_name = chat_dir + chat_files[int(chat_idx) - 1]
                break
            except ValueError:
                print("Invalid input. Please enter a valid integer for the chat record.")

        with open(file_name, "r") as f:
            message = json.load(f)
        self.messages += message
        print(f"LOAD: Chat record '{chat_files[int(chat_idx) - 1][:-5]}' is loaded.")

    async def choose_workflow(self):
        print("\n--------- Setup Adjustment Workflow Options ------------------------------------")
        print("Please enter a request or select an option to guide the AI's tuning process:\n")
        print("  Option 1 (Default): Hotlap/Qualifying Setup**")
        print("  Goal: Maximize single-lap performance and peak grip.\n")
        print("  Option 2: Race Setup Adjustment**")
        print("  Goal: Prioritize consistency and endurance over peak pace.\n")
        print("  Option 3: Custom/Specific Adjustment**")
        print("  Goal: Execute a user requested tuning (assumes user knowledge).\n")
        print("  Option 4: Continue Previous Work**")
        print("  Goal: Resume the sequential setup plan from your last session.\n")
        print("----------------------------------------------------------------------------------\n")

        while True:
            workflow = await asyncio.to_thread(input, "Enter the number of the workflow you wish to start or continue (1-4): ")
            if workflow in ['1', '2', '3', '4']:
                break
            else:
                print("Invalid input. Please enter a valid workflow number (1, 2, 3, or 4).")

        if workflow == '4':
            new_call_message = {
                "role": "user",
                "content": (
                    f"I am leading our previous converstion with **Option {workflow}** workflow. Please continue analyze based on previous conversation.\n"
                ),
            }
            self.messages.append(new_call_message)
            await self.read_message_record()
        
        else:
            new_call_message = {
                "role": "user",
                "content": (
                    f"I am ready to begin the setup process. Please start the analysis using **Option {workflow}** workflow.\n"
                ),
            }
            self.messages.append(new_call_message)


    async def read_new_session(self):
        if self.session is None:
            return "Error: MCP session is not active."
        
        while True:
            try:
                print("\n----------- Loading MoTeC Data ------------------------------------")
                motec_file_info = await self.session.call_tool("get_telemetry_file_info", arguments={})
                print(motec_file_info.content[0].text)
                print("-------------------------------------------------------------------")
                ld_ldx_idx_str = await asyncio.to_thread(input, "Enter the session number you would like to analyze: ")
                ld_ldx_idx = int(ld_ldx_idx_str) - 1
                break 
            except ValueError:
                print("Invalid input. Please enter a valid integer for the session number.")
            except Exception as e:
                print(f"Tool error: {e}")

        while True:
            try:
                print("\n----------- Loading Setup Data ------------------------------------")
                setup_file_info = await self.session.call_tool("get_setup_file_info", arguments={"ld_ldx_idx": ld_ldx_idx})
                print(setup_file_info.content[0].text)
                print("-------------------------------------------------------------------")
                setup_idx_str = await asyncio.to_thread(input, "Enter the setup number you would like to analyze: ")
                setup_idx = int(setup_idx_str) - 1
                break
            except ValueError:
                print("Invalid input. Please enter a valid integer for the setup number.")
            except Exception as e:
                print(f"Tool error: {e}")

        init_seesion = await self.session.call_tool("get_session", arguments={"ld_ldx_idx": ld_ldx_idx, "setup_idx": setup_idx})
        new_call_message = {
            "role": "user",
            "content": (
                "The user uploaded new data collected from a new session for further analysis. This can be initial data or followed up data after following provided recommendation.\n"
            ),
        }
        self.messages.append(new_call_message)
        await pre_flight_setup(self)

        print("\n\n*** New session data has been loaded! ***")

    async def call_tool_directly(self, tool_name: str, arguments: Dict[str, Any] = {}) -> str:
        """
        Executes a specific MCP tool call and adds the result to the history.

        Args:
            tool_name: The name of the tool to call.
            arguments: A dictionary of arguments for the tool.

        Returns:
            The text content returned by the tool.
        """
        if self.session is None:
            return "Error: MCP session is not active."
            
        result = await self.session.call_tool(tool_name, arguments=arguments)
        tool_content = result.content[0].text
        
        pre_call_message = {
            "role": "user",
            "content": (
                f"[PRE-LOADED DATA]: The result of calling the {tool_name} tool."
                f"Use this JSON content in your subsequent analysis:\n---\n{tool_content}\n---"
            ),
        }
        self.messages.append(pre_call_message)
        
        return tool_content


    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available MCP tools.

        Args:
            query: The user query.

        Returns:
            The response from OpenAI.
        """
        self.messages.append({"role": "user", "content": query})
        tools = await self.get_mcp_tools()

        # init API call with conversation history
        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=tools,
            tool_choice="auto",
        )

        # get assistant's response
        assistant_message = response.choices[0].message
        self.messages.append(assistant_message)

        # handle tool calls if present
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                result = await self.session.call_tool(
                    tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )

                # print(f"Calling {tool_call.function.name} tool...")

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result.content[0].text,
                }
                self.messages.append(tool_message)

            #  force loop for non-empty response
            MAX_RETRIES = 3
            final_content = ""
            for retry_count in range(MAX_RETRIES):
                final_response = await self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=tools,
                    tool_choice="auto",
                )

                final_content = final_response.choices[0].message.content
                if final_content and final_content.strip():
                    break  # Exit loop if content is not empty
                
                self.messages.append({
                    "role": "user",
                    "content": "The last response was empty. Please re-evaluate the provided tool output and provide the complete analysis and setup suggestion now. Do not return empty content."
                })

            if not final_content or not final_content.strip():
                final_content = "Analysis failed to produce a response after multiple attempts. Please try again."

            self.messages.append({"role": "assistant", "content": final_content})
            return final_content

        # force loop for no tool call empty response
        MAX_RETRIES = 3
        assistant_content = assistant_message.content
        for retry_count in range(MAX_RETRIES):
            if assistant_content and assistant_content.strip():
                break # exit loop content is not empty

            self.messages.pop() # remove the last empty message
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=self.messages + [{"role": "user", "content": "The last response was empty. Please provide a non-empty response now."}],
                tools=tools,
                tool_choice="none",
            )
            assistant_message = response.choices[0].message
            self.messages.append(assistant_message)
            assistant_content = assistant_message.content

        if not assistant_content or not assistant_content.strip():
            assistant_content = "Analysis failed to produce a response after multiple attempts. Please try again."
        
        return assistant_content

    async def cleanup(self):
        await self.exit_stack.aclose()
        await self.openai_client.close()

async def pre_flight_setup(client: OpenRouterClient):
    try:
        await client.call_tool_directly(
            tool_name="init_session_info", 
        )

        await client.call_tool_directly(
            tool_name="read_processed_lap_time", 
        )
        
    except Exception as e:
        print(f"WARNING: Pre-flight tool call failed (check tool name/arguments): {e}")


async def run_chat_loop(client: OpenRouterClient):
    
    print("\n****** Starting Setup Analysis ******")
    
    await client.choose_workflow()
    await client.read_new_session()

    print("\n\n\n\n------------ Interaction Guide --------------------------------------------------------------------------------------------")
    print("1. Load the session and setup data you want to analyze.")
    print("2. Choose the workflow that best suits your purpose.")
    print("3. Provide the AI setup engineer with feedback on the car’s behavior.")
    print("   Examples: 'The car still oversteers on exit.' or 'Turn 3 entry understeer is too strong.'")
    print("4. After the AI suggests a setup change (e.g., 'Front ARB +1'), test the new setup on track.")
    print("5. Type 'read' or 'r' to load the new test data and setup.")
    print("6. Provide updated feedback to the AI setup engineer based on the latest test results.")
    print("7. When finished, type 'exit'/'e' or 'quit'/'q' to end the session.")
    print("You can always save your current chat and continue later!")
    print("-----------------------------------------------------------------------------------------------------------------------------")
    print("\n****** We recommend having at least 5 valid laps of data to ensure accurate setup recommendations. ******")

    while True:
        try:
            query = await asyncio.to_thread(input, "\n[User]: ")
        except EOFError:
            print("\nExiting chat loop...")
            break
        
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting chat loop...")
            save = input("Would you like to save chat records to continue this session again later? (Y/N) ")
            if not save.lower() in ['n', 'no']:
                await client.save_message_record()
            break

        elif query.lower() == ["read", "r"]:
            client.read_new_session()
        
        if not query.strip():
            continue

        try:
            response = await client.process_query(query)
            if response == "":
                response = await client.process_query(query)

            print(f"[Assistant]: {response}")
        
        except RateLimitError as e:
            print("\n****** RATE LIMIT EXCEEDED ******")
            print(e)
            print("Automatically saving chat record to preserve session history.\n\n")
            await client.save_message_record()
            break

        except Exception as e:
            print(f"An error occurred: {e}")
            break

async def main():
    """Main entry point for the client."""
    client = OpenRouterClient()
    await client.connect_to_server("telemetry_server.py")
    print("Connection established to telemetry server.")

    try:
        await run_chat_loop(client)

    finally:
        print("\nCleaning up resources...")
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())