from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import datetime
import logging
import time
import requests
import threading
import json
import re
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get today's date for the report
current_date = datetime.datetime.now().strftime("%B %d, %Y")

# Templates are kept as is (healthcare_template, finance_template, generic_template)
healthcare_template = """
You are HealthTrendAnalyst, an AI expert specializing in healthcare industry analysis for startups and investors.

### SOURCE CONTENT:
{dom_content}

### ANALYSIS PARAMETERS:
- Industry: Healthcare/Medical
- Analysis Type: {analysis_type}
- Time Focus: {time_period}
- Detail Level: {detail_level}
- Specific Focus: {custom_prompt}

### TASK:
Analyze the provided healthcare industry content and extract relevant market trends, opportunities, regulatory developments, and insights for healthcare startups based on the parameters above.

### OUTPUT FORMAT:
Provide a well-structured markdown analysis with these sections:
1. **Key Market Trends**: Identify 3-5 major healthcare market trends relevant to startups
2. **Emerging Technologies**: Highlight healthcare technologies gaining traction (AI diagnostics, telehealth, wearables, etc.)
3. **Regulatory Landscape**: Summarize recent or upcoming regulatory changes affecting healthcare startups
4. **Funding Environment**: Note any funding trends, significant investments, or active investors in healthcare
5. **Competitive Analysis**: Identify key players, recent acquisitions, and market positioning strategies
6. **Market Opportunities**: Highlight unmet needs and opportunities for healthcare startups
7. **Strategic Recommendations**: Provide actionable insights for healthcare startups

### VISUALIZATION DATA:
Additionally, for visualization purposes, include a JSON block at the end of your response between triple backticks with this structure:
```json
{{
          "market_trends": [
            {{"trend": "Trend Name 1", "impact_score": 85}},
            {{"trend": "Trend Name 2", "impact_score": 75}},
            {{"trend": "Trend Name 3", "impact_score": 65}},
            {{"trend": "Trend Name 4", "impact_score": 60}},
            {{"trend": "Trend Name 5", "impact_score": 55}}
          ],
          "emerging_technologies": [
            {{"technology": "Technology 1", "adoption_rate": 80}},
            {{"technology": "Technology 2", "adoption_rate": 65}},
            {{"technology": "Technology 3", "adoption_rate": 50}},
            {{"technology": "Technology 4", "adoption_rate": 45}},
            {{"technology": "Technology 5", "adoption_rate": 40}}
          ],
          "funding_distribution": [
            {{"sector": "Sector 1", "percentage": 35}},
            {{"sector": "Sector 2", "percentage": 25}},
            {{"sector": "Sector 3", "percentage": 20}},
            {{"sector": "Sector 4", "percentage": 15}},
            {{"sector": "Other", "percentage": 5}}
          ]
        }}
```
Use realistic values based on your analysis. This JSON will be used to create visualizations.

### GUIDELINES:
- Focus on factual information extracted from the source content
- Prioritize recent developments and forward-looking insights
- Be specific about healthcare subsectors (digital health, biotech, medical devices, etc.)
- Include specific companies, technologies, or regulatory bodies mentioned in the content
- Mention specific diseases, conditions, or healthcare challenges when relevant
- If certain sections cannot be addressed from the content, indicate "Insufficient data" rather than inventing information
- For each insight, briefly mention the supporting evidence from the source
- The level of detail should match the requested detail level: {detail_level}
"""

# Finance template remains unchanged
finance_template = """
You are FinTechAnalyst, an AI expert specializing in financial services and fintech industry analysis for startups and investors.

### SOURCE CONTENT:
{dom_content}

### ANALYSIS PARAMETERS:
- Industry: Finance/Fintech
- Analysis Type: {analysis_type}
- Time Focus: {time_period}
- Detail Level: {detail_level}
- Specific Focus: {custom_prompt}

### TASK:
Analyze the provided finance industry content and extract relevant market trends, opportunities, regulatory developments, and insights for fintech startups based on the parameters above.

### OUTPUT FORMAT:
Provide a well-structured markdown analysis with these sections:
1. **Key Market Trends**: Identify 3-5 major fintech market trends relevant to startups
2. **Emerging Technologies**: Highlight financial technologies gaining traction (blockchain, embedded finance, AI, etc.)
3. **Regulatory Landscape**: Summarize recent or upcoming regulatory changes affecting fintech startups
4. **Funding Environment**: Note any funding trends, significant investments, or active investors in fintech
5. **Competitive Analysis**: Identify key players, recent acquisitions, and market positioning strategies
6. **Market Opportunities**: Highlight unmet needs and opportunities for fintech startups
7. **Strategic Recommendations**: Provide actionable insights for fintech startups

### VISUALIZATION DATA:
Additionally, for visualization purposes, include a JSON block at the end of your response between triple backticks with this structure:
```json
{{
          "market_trends": [
            {{"trend": "Trend Name 1", "impact_score": 85}},
            {{"trend": "Trend Name 2", "impact_score": 75}},
            {{"trend": "Trend Name 3", "impact_score": 65}},
            {{"trend": "Trend Name 4", "impact_score": 60}},
            {{"trend": "Trend Name 5", "impact_score": 55}}
          ],
          "emerging_technologies": [
            {{"technology": "Technology 1", "adoption_rate": 80}},
            {{"technology": "Technology 2", "adoption_rate": 65}},
            {{"technology": "Technology 3", "adoption_rate": 50}},
            {{"technology": "Technology 4", "adoption_rate": 45}},
            {{"technology": "Technology 5", "adoption_rate": 40}}
          ],
          "funding_distribution": [
            {{"sector": "Sector 1", "percentage": 35}},
            {{"sector": "Sector 2", "percentage": 25}},
            {{"sector": "Sector 3", "percentage": 20}},
            {{"sector": "Sector 4", "percentage": 15}},
            {{"sector": "Other", "percentage": 5}}
          ]
        }}
```
Use realistic values based on your analysis. This JSON will be used to create visualizations.

### GUIDELINES:
- Focus on factual information extracted from the source content
- Prioritize recent developments and forward-looking insights
- Be specific about fintech subsectors (payments, lending, wealth management, insurance tech, etc.)
- Include specific companies, technologies, or regulatory bodies mentioned in the content
- Mention specific financial products, services, or market segments when relevant
- If certain sections cannot be addressed from the content, indicate "Insufficient data" rather than inventing information
- For each insight, briefly mention the supporting evidence from the source
- The level of detail should match the requested detail level: {detail_level}
"""

# Generic template remains unchanged
generic_template = """
You are IndustryAnalyst, an AI expert specializing in {industry} industry analysis for startups and investors.

### SOURCE CONTENT:
{dom_content}

### ANALYSIS PARAMETERS:
- Industry: {industry}
- Analysis Type: {analysis_type}
- Time Focus: {time_period} 
- Detail Level: {detail_level}
- Specific Focus: {custom_prompt}

### TASK:
Analyze the provided {industry} industry content and extract relevant market trends, opportunities, regulatory developments, and insights for {industry} startups based on the parameters above.

### OUTPUT FORMAT:
Provide a well-structured markdown analysis with these sections:
1. **Key Market Trends**: Identify 3-5 major market trends relevant to startups in this industry
2. **Emerging Technologies**: Highlight technologies gaining traction in the {industry} sector
3. **Regulatory Landscape**: Summarize recent or upcoming regulatory changes affecting {industry} startups
4. **Funding Environment**: Note any funding trends, significant investments, or active investors in {industry}
5. **Competitive Analysis**: Identify key players, recent acquisitions, and market positioning strategies
6. **Market Opportunities**: Highlight unmet needs and opportunities for {industry} startups  
7. **Strategic Recommendations**: Provide actionable insights for {industry} startups

### VISUALIZATION DATA:
Additionally, for visualization purposes, include a JSON block at the end of your response between triple backticks with this structure:
```json
{{
          "market_trends": [
            {{"trend": "Trend Name 1", "impact_score": 85}},
            {{"trend": "Trend Name 2", "impact_score": 75}},
            {{"trend": "Trend Name 3", "impact_score": 65}},
            {{"trend": "Trend Name 4", "impact_score": 60}},
            {{"trend": "Trend Name 5", "impact_score": 55}}
          ],
          "emerging_technologies": [
            {{"technology": "Technology 1", "adoption_rate": 80}},
            {{"technology": "Technology 2", "adoption_rate": 65}},
            {{"technology": "Technology 3", "adoption_rate": 50}},
            {{"technology": "Technology 4", "adoption_rate": 45}},
            {{"technology": "Technology 5", "adoption_rate": 40}}
          ],
          "funding_distribution": [
            {{"sector": "Sector 1", "percentage": 35}},
            {{"sector": "Sector 2", "percentage": 25}},
            {{"sector": "Sector 3", "percentage": 20}},
            {{"sector": "Sector 4", "percentage": 15}},
            {{"sector": "Other", "percentage": 5}}
          ]
        }}
```
Use realistic values based on your analysis. This JSON will be used to create visualizations.

### GUIDELINES:
- Focus on factual information extracted from the source content
- Prioritize recent developments and forward-looking insights  
- Be specific about {industry} subsectors when possible
- Include specific companies, technologies, or regulatory bodies mentioned in the content
- If certain sections cannot be addressed from the content, indicate "Insufficient data" rather than inventing information
- For each insight, briefly mention the supporting evidence from the source
- The level of detail should match the requested detail level: {detail_level}
"""

class TimeoutException(Exception):
    pass

# Cross-platform timeout implementation
class Timeout:
    def __init__(self, seconds):
        self.seconds = seconds
        self.timeout_happened = False
        self._timer = None
        
    def _timeout_function(self):
        self.timeout_happened = True
        
    def __enter__(self):
        self._timer = threading.Timer(self.seconds, self._timeout_function)
        self._timer.daemon = True
        self._timer.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._timer:
            self._timer.cancel()

def create_ollama_model(model_name="llama3:latest", retries=3, backoff=2):
    """Create OllamaLLM model with retry logic"""
    for attempt in range(retries):
        try:
            # Try with different models if available
            model_options = [model_name, "llama3:latest", "llama2:latest", "mistral:latest"]
            
            for model_name in model_options:
                try:
                    logger.info(f"Attempting to initialize Ollama with model: {model_name}")
                    
                    # Check if Ollama server is running
                    try:
                        response = requests.get("http://localhost:11434/api/tags", timeout=5)
                        if response.status_code == 200:
                            available_models = [model["name"] for model in response.json().get("models", [])]
                            logger.info(f"Available Ollama models: {available_models}")
                            
                            if model_name.split(':')[0] not in [m.split(':')[0] for m in available_models]:
                                logger.warning(f"Model {model_name} not found in available models")
                                continue
                        else:
                            logger.warning(f"Ollama API returned status code {response.status_code}")
                            raise Exception("Ollama API is not responding correctly")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"Cannot connect to Ollama server: {str(e)}")
                        raise Exception("Cannot connect to Ollama server - please ensure it's running")
                    
                    # Initialize the model
                    model = OllamaLLM(model=model_name)
                    
                    # Test the model with a simple prompt
                    test_result = model.invoke("Hello")
                    if test_result and len(test_result) > 0:
                        logger.info(f"Successfully initialized Ollama with model: {model_name}")
                        return model
                    else:
                        logger.warning(f"Model {model_name} returned empty response")
                        continue
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize model {model_name}: {str(e)}")
                    continue
            
            raise Exception("All model options failed. Check if Ollama is running and has models installed.")
            
        except Exception as e:
            if attempt < retries - 1:
                sleep_time = backoff ** attempt
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Failed to initialize Ollama after {retries} attempts")
                raise

def extract_visualization_data(text):
    """Extract JSON visualization data from the LLM response with improved error handling"""
    try:
        # Find JSON data between triple backticks with json label
        json_pattern = r"```json\n([\s\S]*?)\n```"
        match = re.search(json_pattern, text)
        
        if match:
            json_str = match.group(1)
            # Strip any non-JSON characters that might cause issues
            json_str = json_str.strip()
            try:
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError as je:
                logger.warning(f"JSON decode error: {str(je)}")
                # Try to fix common JSON issues
                fixed_json = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                fixed_json = re.sub(r',\s*]', ']', fixed_json)  # Remove trailing commas in arrays
                data = json.loads(fixed_json)
                return data
        else:
            # Try without json label
            json_pattern = r"```\n([\s\S]*?)\n```"
            match = re.search(json_pattern, text)
            if match:
                json_str = match.group(1)
                # Check if this looks like JSON
                if '{' in json_str and '}' in json_str:
                    json_str = json_str.strip()
                    try:
                        data = json.loads(json_str)
                        return data
                    except json.JSONDecodeError:
                        # Try more aggressive JSON fixing
                        # Replace single quotes with double quotes
                        fixed_json = json_str.replace("'", '"')
                        # Ensure property names have double quotes
                        fixed_json = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed_json)
                        try:
                            data = json.loads(fixed_json)
                            return data
                        except json.JSONDecodeError:
                            logger.warning("Failed to fix malformed JSON")
                            pass
        
        # As a last resort, try to extract any JSON-like structure
        try:
            import ast
            # Find anything between curly braces
            json_pattern = r"\{[\s\S]*?\}"
            match = re.search(json_pattern, text)
            if match:
                json_str = match.group(0)
                # Try to parse as Python dict and convert to JSON
                py_dict = ast.literal_eval(json_str)
                return json.loads(json.dumps(py_dict))
        except:
            pass
            
        return None
    except Exception as e:
        logger.warning(f"Error extracting visualization data: {str(e)}")
        return None
def debug_json_extraction(text):
    """Debug function to help diagnose JSON extraction issues"""
    logging.info("Debug JSON extraction started")
    
    # Try to find JSON blocks
    json_patterns = [
        r"```json\n([\s\S]*?)\n```",  # JSON with label
        r"```\n([\s\S]*?)\n```",       # Code block without label
        r"\{[\s\S]*?\}"                # Any JSON-like structure
    ]
    
    results = []
    for i, pattern in enumerate(json_patterns):
        matches = re.findall(pattern, text)
        for j, match in enumerate(matches):
            logging.info(f"Pattern {i+1}, Match {j+1}:")
            logging.info(f"Found text: {match[:100]}...")
            
            try:
                # Try to parse as JSON
                data = json.loads(match)
                logging.info("Successfully parsed as JSON!")
                results.append({
                    "pattern": i+1,
                    "match": j+1,
                    "success": True,
                    "data": data
                })
            except json.JSONDecodeError as e:
                logging.info(f"Failed to parse as JSON: {str(e)}")
                results.append({
                    "pattern": i+1,
                    "match": j+1,
                    "success": False,
                    "error": str(e)
                })
    
    return results
def generate_visualizations(data, industry, output_dir="visualizations"):
    """Generate visualizations from the extracted data with improved error handling"""
    if not data:
        logger.warning("No visualization data provided")
        return []
    
    # Create the output directory with an absolute path
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    visualization_paths = []
    industry_colors = {
        "Healthcare": {"primary": "#3498db", "secondary": "#2980b9"},
        "Finance": {"primary": "#2ecc71", "secondary": "#27ae60"},
        "Technology": {"primary": "#9b59b6", "secondary": "#8e44ad"},
        "Retail": {"primary": "#e74c3c", "secondary": "#c0392b"},
        "Manufacturing": {"primary": "#f39c12", "secondary": "#d35400"},
        "default": {"primary": "#34495e", "secondary": "#2c3e50"}
    }
    
    colors = industry_colors.get(industry, industry_colors["default"])
    custom_cmap = LinearSegmentedColormap.from_list("custom", [colors["secondary"], colors["primary"]])
    
    try:
        # Market Trends Bar Chart
        if "market_trends" in data and data["market_trends"]:
            try:
                plt.figure(figsize=(10, 6))
                trends = [item.get("trend", f"Trend {i+1}") for i, item in enumerate(data["market_trends"])]
                impact_scores = []
                
                # Ensure all impact scores are numeric
                for item in data["market_trends"]:
                    score = item.get("impact_score", 50)
                    try:
                        impact_scores.append(float(score))
                    except (ValueError, TypeError):
                        impact_scores.append(50)  # Default value
                
                # Ensure we have data to plot
                if len(trends) > 0 and len(impact_scores) > 0:
                    # Sort by impact score
                    sorted_indices = sorted(range(len(impact_scores)), key=lambda i: impact_scores[i])
                    trends = [trends[i] for i in sorted_indices]
                    impact_scores = [impact_scores[i] for i in sorted_indices]
                    
                    plt.barh(trends, impact_scores, color=colors["primary"])
                    plt.xlabel('Impact Score')
                    plt.title(f'Key {industry} Market Trends by Impact')
                    plt.tight_layout()
                    
                    market_trends_path = os.path.join(output_dir, f"{industry.lower()}_market_trends.png")
                    plt.savefig(market_trends_path)
                    plt.close()
                    visualization_paths.append(market_trends_path)
                    logger.info(f"Created market trends visualization: {market_trends_path}")
                else:
                    logger.warning("Insufficient data for market trends visualization")
                    plt.close()
            except Exception as e:
                logger.error(f"Error creating market trends chart: {str(e)}")
                plt.close()
        
        # Technologies Bar Chart
        if "emerging_technologies" in data and data["emerging_technologies"]:
            try:
                plt.figure(figsize=(10, 6))
                technologies = [item.get("technology", f"Tech {i+1}") for i, item in enumerate(data["emerging_technologies"])]
                adoption_rates = []
                
                # Ensure all adoption rates are numeric
                for item in data["emerging_technologies"]:
                    rate = item.get("adoption_rate", 50)
                    try:
                        adoption_rates.append(float(rate))
                    except (ValueError, TypeError):
                        adoption_rates.append(50)  # Default value
                
                # Ensure we have data to plot
                if len(technologies) > 0 and len(adoption_rates) > 0:
                    sorted_indices = sorted(range(len(adoption_rates)), key=lambda i: adoption_rates[i])
                    technologies = [technologies[i] for i in sorted_indices]
                    adoption_rates = [adoption_rates[i] for i in sorted_indices]
                    
                    plt.barh(technologies, adoption_rates, color=colors["secondary"])
                    plt.xlabel('Adoption Rate (%)')
                    plt.title(f'Emerging Technologies in {industry}')
                    plt.tight_layout()
                    
                    tech_path = os.path.join(output_dir, f"{industry.lower()}_technologies.png")
                    plt.savefig(tech_path)
                    plt.close()
                    visualization_paths.append(tech_path)
                    logger.info(f"Created technologies visualization: {tech_path}")
                else:
                    logger.warning("Insufficient data for technologies visualization")
                    plt.close()
            except Exception as e:
                logger.error(f"Error creating technologies chart: {str(e)}")
                plt.close()
        
        # Funding Pie Chart
        if "funding_distribution" in data and data["funding_distribution"]:
            try:
                plt.figure(figsize=(10, 8))
                sectors = [item.get("sector", f"Sector {i+1}") for i, item in enumerate(data["funding_distribution"])]
                percentages = []
                
                # Ensure all percentages are numeric
                for item in data["funding_distribution"]:
                    pct = item.get("percentage", 20)
                    try:
                        percentages.append(float(pct))
                    except (ValueError, TypeError):
                        percentages.append(20)  # Default value
                
                # Ensure we have data and percentages sum to 100
                if len(sectors) > 0 and len(percentages) > 0:
                    # Normalize percentages to sum to 100
                    total = sum(percentages)
                    if total > 0:
                        percentages = [p * 100 / total for p in percentages]
                        
                    # Create color array of appropriate length
                    color_list = plt.cm.tab10.colors
                    if len(sectors) > len(color_list):
                        # Repeat colors if needed
                        color_list = color_list * (len(sectors) // len(color_list) + 1)
                    colors_to_use = color_list[:len(sectors)]
                    
                    plt.pie(percentages, labels=sectors, autopct='%1.1f%%', startangle=90, 
                            shadow=True, explode=[0.05] * len(sectors),
                            colors=colors_to_use)
                    plt.axis('equal')
                    plt.title(f'{industry} Funding Distribution by Sector')
                    plt.tight_layout()
                    
                    funding_path = os.path.join(output_dir, f"{industry.lower()}_funding.png")
                    plt.savefig(funding_path)
                    plt.close()
                    visualization_paths.append(funding_path)
                    logger.info(f"Created funding visualization: {funding_path}")
                else:
                    logger.warning("Insufficient data for funding visualization")
                    plt.close()
            except Exception as e:
                logger.error(f"Error creating funding chart: {str(e)}")
                plt.close()
            
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
    
    return visualization_paths
def clean_analysis_text(text):
    """Remove JSON data from the analysis text to keep only the report content"""
    # Remove JSON code blocks
    cleaned_text = re.sub(r"```json\n[\s\S]*?\n```", "", text)
    cleaned_text = re.sub(r"```\n[\s\S]*?\n```", "", cleaned_text)
    # Convert < and > to HTML entities to prevent HTML interpretation issues
    cleaned_text = cleaned_text.replace("<", "&lt;").replace(">", "&gt;")
    return cleaned_text

def analyze_trends_with_ollama(dom_chunks, industry, analysis_type, time_period, detail_level, model="llama3:latest", timeout=180, custom_prompt=""):
    """Analyze industry trends from content chunks using Ollama LLM"""
    analysis_params = {
        "industry": industry,
        "analysis_type": analysis_type,
        "time_period": time_period,
        "detail_level": detail_level,
        "custom_prompt": custom_prompt if custom_prompt else ""
    }
    
    # Select template based on industry
    if industry == "Healthcare":
        template = healthcare_template
    elif industry == "Finance":
        template = finance_template
    else:
        template = generic_template
    
    prompt = ChatPromptTemplate.from_template(template)
    
    try:
        # Check if Ollama server is running
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception(f"Ollama API returned status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            error_msg = f"""
# Error Connecting to Ollama Server

The application couldn't connect to the Ollama server. Error: {str(e)}

## Possible solutions:
1. Ensure Ollama is installed - visit https://ollama.ai/download
2. Start the Ollama service with `ollama serve` in a terminal
3. Verify no firewall is blocking port 11434

## Troubleshooting Steps:
1. Open a terminal and run: `curl http://localhost:11434/api/tags`
2. If it returns a list of models, Ollama is running but may not have the required models
3. Run: `ollama pull llama3` to download a model

## Temporary Analysis
Here's a placeholder analysis based on the parameters you provided. Please retry once Ollama is working properly.

### Key Market Trends
- Unable to analyze data due to Ollama server connection error
- Check the extracted content for insights manually
"""
            return {"text": error_msg, "visualizations": []}
            
        model_obj = create_ollama_model(model_name=model)
        chain = prompt | model_obj
    except Exception as e:
        error_msg = f"""
# Error Initializing AI Model

Unfortunately, there was an error initializing the local Ollama model: {str(e)}

## Possible solutions:
1. Ensure Ollama is running properly with `ollama serve` command
2. Check if you have the required models installed with `ollama list`
3. Install needed models with `ollama pull llama3` or `ollama pull mistral`
"""
        return {"text": error_msg, "visualizations": []}
    
    analysis_results = []
    visualization_data_list = []
    
    for i, chunk in enumerate(dom_chunks, start=1):
        logger.info(f"Analyzing chunk {i} of {len(dom_chunks)}")
        
        try:
            result_container = [None]
            exception_container = [None]
            
            def analyze_with_timeout():
                try:
                    # Create the correct input parameters based on template
                    if industry == "Healthcare" or industry == "Finance":
                        invoke_params = {
                            "dom_content": chunk, 
                            "analysis_type": analysis_type,
                            "time_period": time_period,
                            "detail_level": detail_level,
                            "custom_prompt": custom_prompt
                        }
                    else:
                        invoke_params = {
                            "dom_content": chunk, 
                            "industry": industry,
                            "analysis_type": analysis_type,
                            "time_period": time_period,
                            "detail_level": detail_level,
                            "custom_prompt": custom_prompt
                        }
                    
                    result_container[0] = chain.invoke(invoke_params)
                except Exception as e:
                    exception_container[0] = e
            
            analysis_thread = threading.Thread(target=analyze_with_timeout)
            analysis_thread.daemon = True
            analysis_thread.start()
            analysis_thread.join(timeout)
            
            if analysis_thread.is_alive():
                logger.error(f"Analysis of chunk {i} timed out after {timeout} seconds")
                error_response = f"## Analysis Timeout for Content Chunk {i}\n\nThe analysis took too long to complete (timeout after {timeout} seconds)."
                analysis_results.append(error_response)
                continue
            
            if exception_container[0] is not None:
                raise exception_container[0]
            
            response = result_container[0]
            viz_data = extract_visualization_data(response)
            if viz_data:
                visualization_data_list.append(viz_data)
            
            cleaned_response = clean_analysis_text(response)
            analysis_results.append(cleaned_response)
                
        except Exception as e:
            logger.error(f"Error analyzing chunk {i}: {str(e)}")
            error_response = f"## Error Analyzing Content Chunk {i}\n\nThere was an error processing this section: {str(e)}"
            analysis_results.append(error_response)
    
    combined_analysis = "\n\n".join(analysis_results)
    
    # For multiple chunks, add consolidation
    if len(dom_chunks) > 1 and any([not (isinstance(result, str) and result.startswith('## Error')) for result in analysis_results]):
        consolidation_prompt_template = """
        You are a senior industry analyst specializing in {industry} markets.
        
        Below are separate analyses of different content chunks from {industry} industry sources. 
        Consolidate them into a single coherent analysis, removing duplicates and highlighting 
        the most significant insights:
        
        {combined_analysis}
        
        Create a comprehensive {industry} industry report with the following sections:
        
        1. **Executive Summary**: Brief overview of the major findings (2-3 paragraphs)
        
        2. **Key Market Trends**: The most significant trends identified across all analyses
        
        3. **Emerging Technologies**: The most promising technologies for {industry} startups
        
        4. **Regulatory Landscape**: Key regulatory developments affecting the {industry} space
        
        5. **Funding Environment**: Notable funding trends and significant investments
        
        6. **Competitive Analysis**: Key players and competitive dynamics
        
        7. **Market Opportunities**: The best opportunities identified for {industry} startups
        
        8. **Strategic Recommendations**: 5-7 actionable recommendations for {industry} startups
        
        9. **Future Outlook**: Predictions for the next 12-24 months in the {industry} space
           
        ### VISUALIZATION DATA:
        Additionally, for visualization purposes, include a JSON block at the end of your response between triple backticks with this structure:
        ```json
       {{
          "market_trends": [
            {{"trend": "Trend Name 1", "impact_score": 85}},
            {{"trend": "Trend Name 2", "impact_score": 75}},
            {{"trend": "Trend Name 3", "impact_score": 65}},
            {{"trend": "Trend Name 4", "impact_score": 60}},
            {{"trend": "Trend Name 5", "impact_score": 55}}
          ],
          "emerging_technologies": [
            {{"technology": "Technology 1", "adoption_rate": 80}},
            {{"technology": "Technology 2", "adoption_rate": 65}},
            {{"technology": "Technology 3", "adoption_rate": 50}},
            {{"technology": "Technology 4", "adoption_rate": 45}},
            {{"technology": "Technology 5", "adoption_rate": 40}}
          ],
          "funding_distribution": [
            {{"sector": "Sector 1", "percentage": 35}},
            {{"sector": "Sector 2", "percentage": 25}},
            {{"sector": "Sector 3", "percentage": 20}},
            {{"sector": "Sector 4", "percentage": 15}},
            {{"sector": "Other", "percentage": 5}}
          ]
        }}
        ```
        """
        
        try:
            consolidation_prompt = ChatPromptTemplate.from_template(consolidation_prompt_template)
            consolidation_chain = consolidation_prompt | model_obj
            
            result_container = [None]
            exception_container = [None]
            
            def consolidate_with_timeout():
                try:
                    result_container[0] = consolidation_chain.invoke({
                        "combined_analysis": combined_analysis,
                        "industry": industry,
                        "detail_level": detail_level
                    })
                except Exception as e:
                    exception_container[0] = e
            
            consolidation_thread = threading.Thread(target=consolidate_with_timeout)
            consolidation_thread.daemon = True
            consolidation_thread.start()
            consolidation_thread.join(timeout * 2)
            
            if consolidation_thread.is_alive():
                logger.error(f"Consolidation timed out after {timeout*2} seconds")
                return {"text": f"# {industry} Industry Analysis\n\n*Note: Final consolidation could not be completed due to timeout.*\n\n{combined_analysis}", 
                        "visualizations": []}
            
            if exception_container[0] is not None:
                raise exception_container[0]
            
            final_analysis = result_container[0]
            consolidated_viz_data = extract_visualization_data(final_analysis)
            
            visualization_paths = []
            if consolidated_viz_data:
                visualization_paths = generate_visualizations(consolidated_viz_data, industry)
            
            final_text = clean_analysis_text(final_analysis)
            
            return {"text": final_text, "visualizations": visualization_paths}
        
        except Exception as e:
            logger.error(f"Error during consolidation: {str(e)}")
            return {"text": f"# {industry} Industry Analysis\n\n*Error during consolidation: {str(e)}*\n\n{combined_analysis}", 
                    "visualizations": []}
    
    # If no consolidation needed, generate visualizations from individual analyses
    visualization_paths = []
    if visualization_data_list:
        # Handle the first visualization data for simplicity
        # This avoids merging issues and ensures at least one set of visualizations
        if visualization_data_list[0]:
            visualization_paths = generate_visualizations(visualization_data_list[0], industry)
    
    return {"text": combined_analysis, "visualizations": visualization_paths}

def generate_report_with_visuals(analysis_text, visualization_paths, industry, date=None, output_format="html"):
    """Generate a complete report with embedded visualizations"""
    import os
    import logging
    from datetime import datetime
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    # Use current date if not provided
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    logger.debug("Generating {} report for {} industry".format(output_format, industry))
    logger.debug("Received {} visualization paths: {}".format(len(visualization_paths), visualization_paths))
    
    # Validate visualization paths exist
    valid_paths = []
    for path in visualization_paths:
        if os.path.exists(path):
            valid_paths.append(path)
            logger.debug("Verified visualization path exists: {}".format(path))
        else:
            logger.warning("Visualization path does not exist: {}".format(path))
    
    if output_format == "markdown":
        report = "# {} Industry Analysis Report\n\nGenerated on {}\n\n---\n\n{}\n\n".format(
            industry, date, analysis_text
        )
        
        if valid_paths:
            report += "\n## Visualizations\n\n"
            for path in valid_paths:
                filename = os.path.basename(path)
                title = filename.replace("{}_".format(industry.lower()), "").replace(".png", "").replace("_", " ").title()
                report += "### {}\n\n![{}]({})\n\n".format(title, title, path)
        else:
            logger.warning("No valid visualization paths found for markdown report")
        
        return report
    elif output_format == "html":
        try:
            import base64
            
            # Define HTML head
            html_head = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{} Industry Analysis Report</title>
                <meta charset="UTF-8">
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 40px;
                        line-height: 1.6;color: #333;
                        background-color: #f9f9f9;
                        max-width: 1200px;
                        margin: 0 auto;
                    }}
                    h1, h2, h3, h4, h5, h6 {{
                        color: #2c3e50;
                        margin-top: 20px;
                    }}
                    h1 {{
                        font-size: 32px;
                        border-bottom: 2px solid #eaecef;
                        padding-bottom: 10px;
                        margin-bottom: 20px;
                    }}
                    h2 {{
                        font-size: 24px;
                        border-bottom: 1px solid #eaecef;
                        padding-bottom: 8px;
                        margin-bottom: 16px;
                    }}
                    img {{
                        max-width: 100%;
                        height: auto;
                        margin: 20px 0;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .visualization-container {{
                        margin-bottom: 30px;
                    }}
                    .date {{
                        color: #666;
                        margin-bottom: 20px;
                    }}
                    .content {{
                        background-color: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                    }}
                    code {{
                        font-family: Monaco, monospace;
                        background-color: #f0f0f0;
                        padding: 2px 5px;
                        border-radius: 3px;
                    }}
                    pre {{
                        background-color: #f0f0f0;
                        padding: 15px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }}
                    blockquote {{
                        border-left: 4px solid #ddd;
                        padding-left: 15px;
                        color: #666;
                        margin-left: 0;
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }}
                    th, td {{
                        padding: 12px 15px;
                        border-bottom: 1px solid #ddd;
                    }}
                    th {{
                        background-color: #f2f2f2;
                        font-weight: bold;
                    }}
                    tr:hover {{
                        background-color: #f5f5f5;
                    }}
                </style>
            </head>
            """.format(industry)
            
            # Begin HTML body
            html_body = """
            <body>
                <div class="content">
                    <h1>{} Industry Analysis Report</h1>
                    <p class="date">Generated on {}</p>
                    <hr>
                    
                    <div id="analysis-content">
                        {}
                    </div>
            """.format(industry, date, analysis_text)
            
            # Add visualizations if available
            if valid_paths:
                html_body += """
                    <div id="visualizations">
                        <h2>Visualizations</h2>
                """
                
                for path in valid_paths:
                    try:
                        # Read the image file and encode it as base64
                        with open(path, "rb") as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        
                        filename = os.path.basename(path)
                        title = filename.replace("{}_".format(industry.lower()), "").replace(".png", "").replace("_", " ").title()
                        
                        html_body += """
                        <div class="visualization-container">
                            <h3>{}</h3>
                            <img src="data:image/png;base64,{}" alt="{}">
                        </div>
                        """.format(title, img_data, title)
                        
                        logger.debug("Successfully embedded visualization: {}".format(title))
                    except Exception as e:
                        logger.error("Error embedding visualization {}: {}".format(path, str(e)))
                        html_body += """
                        <div class="visualization-container">
                            <h3>Error Loading Visualization</h3>
                            <p>Could not load: {} (Error: {})</p>
                        </div>
                        """.format(os.path.basename(path), str(e))
                
                html_body += """
                    </div>
                """
            else:
                logger.warning("No visualizations available for HTML report")
            
            # Close HTML document
            html_body += """
                </div>
            </body>
            </html>
            """
            
            return html_head + html_body
            
        except Exception as e:
            logger.error("Error generating HTML report: {}".format(str(e)))
            return """
            <!DOCTYPE html>
            <html>
            <head><title>Error Generating Report</title></head>
            <body>
                <h1>Error Generating {} Industry Report</h1>
                <p>An error occurred: {}</p>
                <pre>{}</pre>
            </body>
            </html>
            """.format(industry, str(e), analysis_text)
    else:
        raise ValueError("Unsupported output format: {}".format(output_format))

def save_report(report_content, industry, output_format="html", output_dir="reports"):
    """Save the generated report to a file"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{industry.lower()}_analysis_{timestamp}.{output_format}"
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    logger.info(f"Report saved to {file_path}")
    return file_path

def analyze_content(content, industry="Technology", analysis_type="Comprehensive", 
                   time_period="Current and Near-Future", detail_level="Detailed", 
                   model="llama3:latest", custom_prompt="", timeout=180):
    """Main function to analyze content and generate a report"""
    
    # Split content into chunks if needed
    max_chunk_length = 4000  # Characters per chunk
    content_chunks = []
    
    if len(content) > max_chunk_length:
        # Simple chunking by character length
        chunks = [content[i:i+max_chunk_length] for i in range(0, len(content), max_chunk_length)]
        content_chunks = chunks
    else:
        content_chunks = [content]
    
    # Log analysis parameters
    logger.info(f"Analyzing {industry} industry content ({len(content_chunks)} chunks)")
    logger.info(f"Parameters: {analysis_type} analysis, {time_period} focus, {detail_level} detail")
    
    # Analyze trends
    analysis_result = analyze_trends_with_ollama(
        content_chunks, 
        industry, 
        analysis_type, 
        time_period, 
        detail_level,
        model=model,
        timeout=timeout,
        custom_prompt=custom_prompt
    )
    
    # Generate report with visualizations
    report_content = generate_report_with_visuals(
        analysis_result["text"],
        analysis_result["visualizations"],
        industry,
        date=current_date
    )
    
    # Save report to file
    report_path = save_report(report_content, industry)
    
    return {
        "report_content": report_content,
        "report_path": report_path,
        "visualization_paths": analysis_result["visualizations"]
    }

def main():
    """Command line interface for the analysis tool"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Industry Analysis Tool using Ollama")
    parser.add_argument("--file", type=str, help="Path to a text file containing content to analyze")
    parser.add_argument("--industry", type=str, default="Technology", 
                       choices=["Technology", "Healthcare", "Finance", "Retail", "Manufacturing", "Other"],
                       help="Industry to analyze")
    parser.add_argument("--analysis-type", type=str, default="Comprehensive",
                       choices=["Comprehensive", "Market Trends", "Technology Focus", "Competitive Analysis", "Investment Opportunities"],
                       help="Type of analysis to perform")
    parser.add_argument("--time-period", type=str, default="Current and Near-Future",
                       choices=["Current", "Current and Near-Future", "Long-term Future"],
                       help="Time period to focus on")
    parser.add_argument("--detail-level", type=str, default="Detailed",
                       choices=["Brief", "Detailed", "Comprehensive"],
                       help="Level of detail in the analysis")
    parser.add_argument("--model", type=str, default="llama3:latest",
                       help="Ollama model to use for analysis")
    parser.add_argument("--custom-prompt", type=str, default="",
                       help="Custom prompt to guide the analysis")
    parser.add_argument("--timeout", type=int, default=180,
                       help="Timeout for each analysis chunk in seconds")
    parser.add_argument("--output-format", type=str, default="html",
                       choices=["html", "markdown"],
                       help="Output format for the report")
    
    args = parser.parse_args()
    
    # Get content from file or stdin
    content = ""
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            sys.exit(1)
    else:
        print("Enter or paste the content to analyze (Ctrl+D or Ctrl+Z to finish):")
        try:
            content = sys.stdin.read()
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(0)
    
    if not content.strip():
        print("Error: No content provided for analysis.")
        sys.exit(1)
    
    # Perform analysis
    try:
        result = analyze_content(
            content,
            industry=args.industry,
            analysis_type=args.analysis_type,
            time_period=args.time_period,
            detail_level=args.detail_level,
            model=args.model,
            custom_prompt=args.custom_prompt,
            timeout=args.timeout
        )
        
        print(f"\nAnalysis complete! Report saved to: {result['report_path']}")
        if result['visualization_paths']:
            print(f"Visualizations generated: {len(result['visualization_paths'])}")
            for path in result['visualization_paths']:
                print(f" - {path}")
    except Exception as e:
        print(f"Error performing analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()