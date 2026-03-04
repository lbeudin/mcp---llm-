
import os
from dotenv import load_dotenv
import asyncio
from agents import Agent, Runner, WebSearchTool, ModelSettings, set_default_openai_api
from agents.mcp import MCPServerStdio
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from openai.types.shared import Reasoning

load_dotenv(override=True) # load the API key from the .env file. We set override to True here to ensure the notebook is loading any changes
set_default_openai_api(os.getenv("OPENAI_API_KEY"))


# draft - not yet finished



async def main() -> None:
    async with MCPServerStdio(
        name="Codex CLI",
        params={
            "command": "npx",
            "args": ["-y", "codex", "mcp-server"],
        },
        client_session_timeout_seconds=360000,
    ) as codex_mcp_server:
        print("Codex MCP server started.")


        feature_agent = Agent(
            name="Feature Engineer",
            instructions=(
                "You are an expert in engineering features, you receive the user request in userRequest.txt"
                " you need to interpret the business intent behind the request and map it to relevant features in the database to build a clustering model."
                "you own many database and the context of each features."
                "Save your work in a json file called pricingFeatureEngineered.json in the current directory."
                "Your responsibilities include:"
                "- Interpret business intent"
                "- Map the request to relevant database features"
                "- Propose engineered features (aggregations, rolling metrics, normalization, volatility measures, behavioral indicators, etc.)."
                "- identify the number of required cluster and the model to use among : ."
                "- Define preprocessing steps (scaling, encoding, missing value handling)."
                "Deliverable will be a json with A1-Answer.json structure and name it Feature_Engineer.json"

                "Always call codex with \"approval-policy\": \"never\" and \"sandbox\": \"workspace-write\""
            ),
            # adding features database + context
            # adding pre processing step + link to python functions
            # adding clustering model selection + link to python functions
            # create the json reader to match the response with algo
            model="gpt-5",
            mcp_servers=[codex_mcp_server],
            #tools=[WebSearchTool()], agent ability to google search
        )



        business_agent = Agent(
            name="Business Interpreter",
            instructions=(
                "You are a cluster validator connoisseur. Come up with an idea for a single page html + css + javascript game that a developer could build in about 50 lines of code. "
                "This agent must:"
                " - Translate clustering results into business insights"
                " - Provide contextual interpretation for each cluster"
                " - Explain quality metrics in business terms."
                "Format should be brief done per each cluster and be saved under a business.txt file. Focus on actionable recommendations for stakeholders based on the clustering outcomes."
            ),
            model="gpt-5",
        )
        quant_agent = Agent(
            name="Quant Interpreter",
            instructions=(
                "You are a cluster validator connoisseur. Come up with an idea for a single page html + css + javascript game that a developer could build in about 50 lines of code. "
                "This agent must:"
                " - Evaluate cluster usefulness and actionability"
                " - Detect weak clustering (low separation, unstable over time)"
                " - Explain quality metrics"
                "Propose if the quality is not reaching the accepted standard :"
                " - Feature improvements"
                " - Additional data sources"
                " - Alternative modeling approaches"
                " - ADimensionality reduction if needed (PCA/UMAP)"
                "Output :"
                " - cluster narratives"
                " - Quality assessment report"
                " - Improvement recommendations"

                "Format should be brief done per each cluster and overall insights and be saved under a quant.txt file. Focus on actionable recommendations for stakeholders based on the clustering outcomes."
            ),
            model="gpt-5",
            handoffs=[business_agent]
        )

        profiling_agent = Agent(
            name="Profiler Interpreter",
            instructions=(
                "You are a behavioral Profiling Agent. From the realized clustering you received the probabilistic breakdown of categorical variables. "
                "This agent must for each cluster:"
                " - Identify dominant behaviors and risk profiles"
                " -  Provide descriptive statistical summary"
                " - Generate interpretable cluster personas"
                " - Quantify uncertainty where relevant"
                "Format should be concise done per each cluster."
            ),
            # execute probability general table on the cluster and use the result for this agent
            model="gpt-5",
            handoffs=[business_agent],
        )

        clustering_agent = Agent(
            name="Cluster executor",
            instructions=(
                "TODO Lea connect to run.cmd"
            ),
            model="gpt-5",
        )


        project_manager_agent = Agent(
            name="Project Manager",
            instructions=(
                f"""{RECOMMENDED_PROMPT_PREFIX}"""
                """
                You are the Project Manager. 
                Objective:
                You receive the Use request and you are responsible for coordinating the team to deliver the project.
                You will work with the feature_agent, quant_agent,profiling_agent, and business_agent to ensure the project is completed successfully.
        
                Deliverables (write in project root):
                - userRequest.txt: the original user request saved as a text file with
                - requestUserProcessed.txt detailing the refined request, requirements, and tasks for each role. This is the main output of your role and should be brief and clear.
                    - Request name
                    - Required deliverables for each roles (exact file names and purpose)
                    - Key technical notes and constraints
    
                Process:
                - Resolve ambiguities with minimal, reasonable assumptions. Be specific so each role can act without guessing.
                - Create files using Codex MCP with {"approval-policy":"never","sandbox":"workspace-write"}.
                - Do not create folders. Only create requestUserProcessed.txt.
    
                Handoffs (gated by required files):
                1) After the three files above are created, hand off to the feature_agent and include userRequest.txt
                2) Wait for the feature_agent to produce Feature_Engineer.json. hand off to the clustering_agent
                3) Wait clustering_agent to produce the statistic.json file. Probability.json 
                4) When both exists, hand off in parallel to both:
                    - quant_agent
                    - profiling_agent
                4) Wait for quant_agent  to produce quant.txt and profiling_agent to profiling.text . Verify both files exist.
                5) When both exist, hand off to the business_agent with profiling_agent and provide all prior artifacts and outputs.
                6) Do not advance to the next handoff until the required files for that step are present. If something is missing, request the owning agent to supply it and re-check.
    
                PM Responsibilities:
                - Coordinate all roles, track file completion, and enforce the above gating checks.
                - Do NOT respond with status updates. Just handoff to the next agent until the project is complete.
                """
            ),
            model="gpt-5",
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="medium")
            ),
            handoffs=[business_agent, profiling_agent, quant_agent, feature_agent],
            mcp_servers=[codex_mcp_server],
        )

        # After constructing the Project Manager, the script sets every specialist's handoffs back to the Project Manager. This ensures deliverables return for validation before moving on.
        business_agent.handoffs = [project_manager_agent]
        profiling_agent.handoffs = [project_manager_agent]
        quant_agent.handoffs = [project_manager_agent]
        feature_agent.handoffs = [project_manager_agent]



        result = await Runner.run(project_manager_agent, "Implement a fun new game!")

        #This is the task that the Project Manager will refine into specific requirements and tasks for the entire system.
        task_list = """
        Goal: Build a clustering multiple reports showcase a multi-agent workflow.

        High-level requirements:
        - Proposing ideal features with pre-process on the data.
        - Cluster the data and produce clustering quality metrics.
        - Produce a behavioral profiling of the clusters.
        - Evaluate the clusters from a quant perspective and propose improvements if needed.
        - Translate the clustering results into business insights and recommendations.

        Roles:
        - feature engineer: interpret the user request, map it to relevant features in the database, propose engineered features
        - quant interpreter: evaluate cluster usefulness and actionability, detect weak clustering, explain quality metrics, propose improvements if needed.
        - profiling interpreter: identify dominant behaviors and risk profiles, provide descriptive statistical summary, generate interpretable cluster personas, quantify uncertainty where relevant.
        - business agent: translate clustering results into business insights, provide contextual interpretation for each cluster, explain quality metrics in business terms.

        Constraints:
        - No external database—memory storage is fine.
        - Keep everything readable for beginners; no frameworks required.
        - All outputs should be small files saved in clearly named folders.
        """

        # Only the Project Manager receives the task list directly
        result = await Runner.run(project_manager_agent, task_list, max_turns=10)
        print(result.final_output)


        return

if __name__ == "__main__":
    # Jupyter/IPython already runs an event loop, so calling asyncio.run() here
    # raises "asyncio.run() cannot be called from a running event loop".
    # Workaround: if a loop is running (notebook), use top-level `await`; otherwise use asyncio.run().
    try:
        asyncio.get_running_loop()
        await main()
    except RuntimeError:
        asyncio.run(main())

#codex() is used for creating a conversation.
#codex-reply() is for continuing a conversation.