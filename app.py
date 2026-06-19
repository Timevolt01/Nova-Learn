from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    redirect,
    url_for,
    send_from_directory,
    session,
)
from markupsafe import Markup
from dotenv import load_dotenv
import os
import markdown
import socket
import json
import requests
import re
import sqlite3
import urllib.parse
import functools

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    LANGCHAIN_AVAILABLE = True
except Exception:
    ChatGoogleGenerativeAI = None
    PromptTemplate = None
    LLMChain = None
    LANGCHAIN_AVAILABLE = False

# Create the templates directory if it doesn't exist
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)

# Initialize Flask app
app = Flask(__name__, static_folder="static")

# Load environment variables
load_dotenv()

# Ensure API keys are available
google_api_key = os.getenv("GOOGLE_API_KEY")
yt_api_key = os.getenv("YT_API_KEY")
app.secret_key = os.getenv("SECRET_KEY", "default-secret-key-for-dev")

if not google_api_key:
    print("GOOGLE_API_KEY not configured; AI roadmap generation will use fallbacks.")

if not yt_api_key:
    print("YT_API_KEY not configured; YouTube integration will use public scraping fallback.")

# Define the prompt template with more explicit instructions for JSON format
PROMPT_TEMPLATE = """ 
Subject: {text_content}
You are an expert in generating roadmaps for {text_content}. 
Create a comprehensive roadmap for learning {text_content} from zero prior knowledge, aiming to reach an intermediate level of proficiency suitable for practical use (e.g., building basic projects, understanding core concepts, applying skills creatively). Include the following: 
Create a comprehensive roadmap for learning {text_content} from zero prior knowledge, aiming to reach an intermediate level of proficiency suitable for practical use.

YOUR RESPONSE MUST BE A VALID JSON LIST OF OBJECTS with the following schema:
[
  {{
    "month": "Month 1",
    "milestone": "Understanding the basics",
    "tasks": [
      {{
        "task": "Learn the fundamentals of X",
        "estimated_time": "4 hours",
        "resources": "Resource 1, Resource 2"
      }},
      {{
        "task": "Practice basic Y concepts",
        "estimated_time": "6 hours",
        "resources": "Resource 3"
      }}
    ]
  }},
  {{
    "month": "Month 2",
    "milestone": "Building on fundamentals",
    "tasks": [
      {{
        "task": "Intermediate concept Z",
        "estimated_time": "5 hours",
        "resources": "Resource 4, Resource 5"
      }}
    ]
  }}
]

Include the following in your roadmap:
- Monthly milestones with specific, beginner-friendly outcomes
- Detailed tasks per month, including foundational learning, hands-on practice, and small projects
- Estimated time commitments for each task (assume 10-15 hours/week available)
- Introduction of key concepts, tools, or techniques gradually, building from basics to intermediate topics
- Curated, beginner-accessible resources (tutorials, books, videos) for each task
- A final mini-project or demonstration of skill with clear completion criteria

 
- Monthly milestones with specific, beginner-friendly outcomes (e.g., 'Write a simple script,' 'Explain Newton’s laws').  
- Detailed tasks per month, including foundational learning, hands-on practice, and small projects, with estimated time commitments (assume 10-15 hours/week).  
- Introduction of key concepts, tools, or techniques gradually, building from basics to intermediate topics (e.g., variables to functions in coding, lines to composition in art).  
- Curated, beginner-accessible resources (e.g., free tutorials, introductory books, videos) emphasizing clarity and engagement.  
- Common beginner challenges (e.g., information overload, lack of confidence) and practical mitigation strategies (e.g., spaced repetition, community support).  
- A final mini-project or demonstration of skill (e.g., a portfolio piece, a solved problem set) with clear completion criteria. 

GIVE ROADMAP IN LIST OF DICTIONRIES FORMAT.
DO NOT INCLUDE ANY OTHER TEXT OR EXPLANATION.
"""


# Quiz template for generating MCQ questions
QUIZ_TEMPLATE = """
Subject: {text_content}
You are an expert in generating MCQ type quiz on the basis of subject. 
Given the above text, create a quiz of 10 multiple choice questions keeping difficulty level mixed some easy some moderate some hard. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide.
Ensure to make an array of 3 MCQs referring to the following response json. IT SHOULD BE IN STRING FORMAT.
Here is the RESPONSE_JSON: 

{response_json}
"""

# Response JSON template for quiz
response_json = {
    "mcqs": [
        {
            "mcq": "multiple choice question1",
            "options": {
                "a": "choice here1",
                "b": "choice here2",
                "c": "choice here3",
                "d": "choice here4",
            },
            "correct": "a",
            "topic": "topic here",
            "difficulty": "difficulty here",
            "explanation": "explanation here for the correct answer",
        },
        {
            "mcq": "multiple choice question1",
            "options": {
                "a": "choice here1",
                "b": "choice here2",
                "c": "choice here3",
                "d": "choice here4",
            },
            "correct": "a",
            "topic": "topic here",
            "difficulty": "difficulty here",
            "explanation": "explanation here for the correct answer",
        },
        {
            "mcq": "multiple choice question1",
            "options": {
                "a": "choice here1",
                "b": "choice here2",
                "c": "choice here3",
                "d": "choice here4",
            },
            "correct": "a",
            "topic": "topic here",
            "difficulty": "difficulty here",
            "explanation": "explanation here for the correct answer",
        },
    ]
}

# Create the PromptTemplate for both roadmap and quiz
if LANGCHAIN_AVAILABLE:
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["text_content"],
    )

    quiz_prompt = PromptTemplate(
        template=QUIZ_TEMPLATE,
        input_variables=["text_content", "response_json"],
    )
else:
    prompt = None
    quiz_prompt = None


def escape_json_string(json_string):
    """Escape problematic characters in a JSON string."""
    return (
        json_string.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def generate_quiz(text_content):
    """Generate a quiz based on the given subject"""
    if not LANGCHAIN_AVAILABLE or not google_api_key:
        return {
            "error": "AI quiz generation is unavailable because the required service configuration is missing."
        }

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Escape the response_json before passing it to the prompt
    escaped_response_json = escape_json_string(json.dumps(response_json, indent=2))

    # Create the LLMChain with the formatted prompt
    name_chain = LLMChain(llm=llm, prompt=quiz_prompt)

    # Use invoke() to get the response
    try:
        response = name_chain.invoke(
            {
                "text_content": text_content,
                "response_json": escaped_response_json,
            }
        )

        # Clean and parse the response
        json_str = response["text"].strip("```json\n").strip("```")
        json_str = json_str.replace("\n", "").replace("\r", "")  # Remove newlines
        extracted_response = json.loads(json_str)

        return extracted_response
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {str(e)}")
        return {"error": f"Invalid JSON response: {str(e)}"}
    except Exception as e:
        print(f"Error generating quiz: {str(e)}")
        return {"error": f"Error generating quiz: {str(e)}"}


# Function to generate roadmap
def generate_roadmap(text_content):
    if not LANGCHAIN_AVAILABLE or not google_api_key:
        return "Error generating roadmap: Google AI credentials are not configured"

    # Initialize the Google Generative AI model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=google_api_key,
        temperature=0,  # Low temperature for structured output
        max_output_tokens=4096,
        timeout=60,
        max_retries=3,
    )

    # Create an LLMChain with the prompt and model
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain with the input subject
    try:
        response = chain.invoke({"text_content": text_content})
        response_text = response["text"]

        # Extract JSON from response
        json_str = response_text.strip()

        # If the response is wrapped in code blocks, remove them
        if json_str.startswith("```json"):
            json_str = json_str.strip("```json\n").rstrip("```")
        elif json_str.startswith("```"):
            json_str = json_str.strip("```\n").rstrip("```")

        # Try to find JSON array in the response using regex
        json_match = re.search(r"(\[\s*\{.*\}\s*\])", json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)

        # Clean up any trailing or leading text that might interfere with JSON parsing
        json_str = json_str.strip()

        # Parse the JSON
        try:
            roadmap_data = json.loads(json_str)

            # Validate the expected structure
            if not isinstance(roadmap_data, list):
                return "Error generating roadmap: Response is not a list"

            for item in roadmap_data:
                if not isinstance(item, dict):
                    return "Error generating roadmap: List items are not dictionaries"
                if "month" not in item:
                    return "Error generating roadmap: Missing 'month' key in response"
                if "tasks" not in item:
                    return "Error generating roadmap: Missing 'tasks' key in response"

            return roadmap_data
        except json.JSONDecodeError as e:
            return f"Error decoding JSON: {str(e)}\nResponse was: {json_str[:100]}..."
    except Exception as e:
        return f"Error generating roadmap: {str(e)}"


def extract_tasks(roadmap):
    """
    Extracts tasks from a roadmap and returns them as search queries
    """
    search_queries = []

    if isinstance(roadmap, str) and roadmap.startswith("Error"):
        return []

    try:
        for month in roadmap:
            for task in month["tasks"]:
                search_queries.append(task["task"])
    except (KeyError, TypeError):
        # Handle case where the structure isn't as expected
        pass

    return search_queries


def search_youtube_videos(query, max_results=5):
    """
    Search YouTube for videos matching the query without using the API.
    Returns a list of videos with title, video ID, and embed code.
    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return
    Returns:
        list: List of dictionaries containing video information
    """
    query = urllib.parse.quote(query)
    url = f"https://www.youtube.com/results?search_query={query}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Extract initial data JSON from the page
        initial_data_pattern = r"var ytInitialData = (.+?);</script>"
        matches = re.search(initial_data_pattern, response.text)

        if not matches:
            return []

        json_str = matches.group(1)
        data = json.loads(json_str)

        videos = []

        # Navigate through the nested JSON structure to find video results
        contents = (
            data.get("contents", {})
            .get("twoColumnSearchResultsRenderer", {})
            .get("primaryContents", {})
            .get("sectionListRenderer", {})
            .get("contents", [{}])[0]
            .get("itemSectionRenderer", {})
            .get("contents", [])
        )

        for item in contents:
            video_renderer = item.get("videoRenderer")
            if not video_renderer:
                continue

            video_id = video_renderer.get("videoId")
            if not video_id:
                continue

            # Get video title
            title_runs = video_renderer.get("title", {}).get("runs", [])
            title = "".join([run.get("text", "") for run in title_runs])

            # Create embed code
            embed_code = f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>'

            # Get thumbnail URL
            thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"

            videos.append(
                {
                    "video_id": video_id,
                    "title": title,
                    "embed_code": embed_code,
                    "thumbnail": thumbnail,
                    "watch_url": f"https://www.youtube.com/watch?v={video_id}",
                }
            )

            if len(videos) >= max_results:
                break

        return videos

    except Exception as e:
        print(f"Error searching YouTube: {str(e)}")
        return []


def get_video_details(video_id):
    """
    Get detailed information about a specific YouTube video.

    Args:
        video_id (str): YouTube video ID

    Returns:
        dict: Video details including title, description, etc.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Extract ytInitialPlayerResponse using a more precise regex
        player_response_pattern = (
            r"ytInitialPlayerResponse\s*=\s*({[^}]*(?:}[^}]*)*})\s*;"
        )
        matches = re.search(player_response_pattern, response.text)

        if not matches:
            print("Could not find ytInitialPlayerResponse in the page")
            return None

        try:
            # Clean the JSON string before parsing
            json_str = matches.group(1)
            # Remove any trailing characters that might interfere with JSON parsing
            json_str = re.sub(r"\s*;.*$", "", json_str)
            # Handle potential line breaks and escape sequences
            json_str = json_str.replace("\n", "").replace("\r", "")

            # Try to find the actual end of the JSON object
            brace_count = 0
            clean_json = ""
            for char in json_str:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                clean_json += char
                if brace_count == 0:
                    break

            data = json.loads(clean_json)

            # Extract video details
            video_details = data.get("videoDetails", {})
            microformat = data.get("microformat", {}).get(
                "playerMicroformatRenderer", {}
            )

            length_seconds = video_details.get("lengthSeconds", "0")
            try:
                length_seconds = int(length_seconds)
            except (ValueError, TypeError):
                length_seconds = 0

            return {
                "video_id": video_id,
                "title": video_details.get("title", ""),
                "description": video_details.get("shortDescription", ""),
                "channel_name": video_details.get("author", ""),
                "lengthSeconds": length_seconds,
                "embed_code": f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>',
                "watch_url": url,
                "thumbnail": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                "publishDate": microformat.get("publishDate", ""),
                "uploadDate": microformat.get("uploadDate", ""),
            }

        except json.JSONDecodeError as json_error:
            print(f"JSON parsing error in clean_json: {str(json_error)}")
            print(
                f"Problematic JSON string: {clean_json[:100]}..."
            )  # Print first 100 chars for debugging
            return None
        except Exception as e:
            print(f"Error processing video details: {str(e)}")
            return None

    except requests.RequestException as e:
        print(f"Error fetching video page: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error in get_video_details: {str(e)}")
        return None


def fetch_youtube_video(query, subject=None):
    """
    Fetches a YouTube video based on a search query and returns an embeddable iframe.
    Now includes the subject in search to ensure relevance and checks for video duration.

    Args:
        query (str): The search query
        subject (str): The main subject of the roadmap

    Returns:
        dict: Dictionary with video_id, title, and embed_code
    """
    try:
        # Include the subject in the search query to keep videos on topic
        search_term = query
        if subject:
            # Make sure subject is actually in the query
            if subject.lower() not in query.lower():
                search_term = f"{subject} {query}"

        # Search for videos based on the enhanced query
        results = search_youtube_videos(f"{search_term} tutorial long", max_results=5)

        # Filter for videos longer than 15 minutes
        suitable_video = None
        for video in results:
            video_details = get_video_details(video["video_id"])
            if video_details:
                try:
                    duration = int(video_details.get("lengthSeconds", 0))
                    if duration > 400:  # 15 minutes = 900 seconds
                        suitable_video = video
                        break
                except (ValueError, TypeError):
                    continue

        if suitable_video:
            return suitable_video
        else:
            return {
                "video_id": None,
                "title": "No suitable video found",
                "embed_code": "No video found for this query.",
            }

    except Exception as e:
        return {
            "video_id": None,
            "title": f"Error: {str(e)}",
            "embed_code": f"Error fetching video: {str(e)}",
        }


def generate_fallback_roadmap(subject):
    """
    Generates a basic roadmap when the AI model fails to produce a valid JSON response
    """
    fallback_roadmap = f"""
# Learning Roadmap for {subject}

## Month 1: Getting Started
- Learn the fundamentals
- Practice basic concepts
- Set up your learning environment

## Month 2: Building Foundation
- Dive deeper into core concepts
- Complete beginner projects
- Review and reinforce what you've learned

## Month 3: Intermediate Skills
- Learn more advanced techniques
- Work on a personal project
- Connect with the community

*Note: This is a basic template. The AI-generated detailed roadmap could not be created. Please try again or modify your query.*
    """
    return fallback_roadmap


def format_roadmap_with_videos(roadmap_data, subject=None):
    """
    Formats the AI-generated roadmap with corresponding YouTube video links.
    Returns a markdown-formatted string.
    """
    if isinstance(roadmap_data, str) and roadmap_data.startswith("Error"):
        # Extract the subject from the error message if possible
        subject_match = re.search(r"for\s+(.+?)(?:\s+from|\s*$)", roadmap_data)
        subject = subject_match.group(1) if subject_match else "your subject"
        return generate_fallback_roadmap(subject)

    formatted_roadmap = "# Learning Roadmap\n\n"

    try:
        for month in roadmap_data:
            formatted_roadmap += f"## {month['month']}\n\n"

            if "milestone" in month:
                formatted_roadmap += f"**Milestone:** {month['milestone']}\n\n"

            for task in month["tasks"]:
                # Fetch a relevant YouTube video for this task, including subject
                video = fetch_youtube_video(f"{task['task']}", subject=subject)

                formatted_roadmap += f"### {task['task']}\n"

                if "estimated_time" in task:
                    formatted_roadmap += (
                        f"- **Estimated Time:** {task['estimated_time']}\n"
                    )

                if "resources" in task:
                    formatted_roadmap += f"- **Resources:** {task['resources']}\n"

                # Add video embed - ensure it's properly formatted for Markdown to HTML conversion
                if video["video_id"]:
                    formatted_roadmap += f"\n**Tutorial Video:** {video['title']}\n\n"
                    # Use HTML directly instead of relying on Markdown conversion for iframes
                    formatted_roadmap += (
                        f"<div class='video-container'>{video['embed_code']}</div>\n\n"
                    )
                else:
                    formatted_roadmap += "\n*No tutorial video found*\n\n"
    except (KeyError, TypeError) as e:
        # If there's any error in the structure, fall back to basic roadmap
        subject = "your subject"
        return generate_fallback_roadmap(subject)

    return formatted_roadmap


def enhance_roadmap_with_videos(roadmap_data, subject=None):
    """
    Enhances the roadmap JSON with YouTube videos and returns both JSON and markdown
    """
    if isinstance(roadmap_data, str) and roadmap_data.startswith("Error"):
        # Create a fallback markdown for error cases
        markdown_version = generate_fallback_roadmap("the requested subject")
        return roadmap_data, markdown_version

    try:
        # Create a deep copy of the roadmap to avoid modifying the original
        enhanced_roadmap = json.loads(json.dumps(roadmap_data))

        # Add videos to each task
        for month in enhanced_roadmap:
            for task in month["tasks"]:
                # Fetch a relevant YouTube video for this task, including subject
                video = fetch_youtube_video(f"{task['task']}", subject=subject)
                task["video"] = video

        # Generate markdown version
        markdown_version = format_roadmap_with_videos(roadmap_data, subject=subject)

        return enhanced_roadmap, markdown_version
    except Exception as e:
        error_msg = f"Error enhancing roadmap: {str(e)}"
        markdown_version = generate_fallback_roadmap("the requested subject")
        return error_msg, markdown_version


# Custom function to convert markdown to HTML
def md_to_html(text):
    try:
        # Convert markdown to HTML, but allow raw HTML to pass through
        return Markup(
            markdown.markdown(text, extensions=["tables", "fenced_code", "extra"])
        )
    except Exception as e:
        # Fall back to preformatted text if there's any issue
        return Markup(f"<pre>{text}</pre>")


def init_db():
    conn = sqlite3.connect("instance/roadmaps.db")
    cursor = conn.cursor()

    # Create table for storing roadmaps
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS roadmaps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        user_id TEXT DEFAULT 'anonymous'
    )
    """
    )

    conn.commit()
    conn.close()

    print("Database initialized successfully!")


def init_user_db():
    """Initialize the users database table"""
    conn = sqlite3.connect("instance/roadmaps.db")
    cursor = conn.cursor()

    # Create table for storing users
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    )

    conn.commit()
    conn.close()

    print("User database initialized successfully!")


# Register filter with Jinja


def hash_password(password):
    """Hash the password for secure storage"""
    # In a production app, you'd use a library like bcrypt
    # This is a simple hashing for demonstration
    import hashlib

    return hashlib.sha256(password.encode()).hexdigest()


def register_user(name, email, password):
    """Register a new user in the database"""
    conn = sqlite3.connect("instance/roadmaps.db")
    cursor = conn.cursor()

    try:
        # Hash the password before storing
        hashed_password = hash_password(password)

        # Insert the user into the database
        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name, email, hashed_password),
        )

        conn.commit()
        user_id = cursor.lastrowid
        return user_id
    except sqlite3.IntegrityError:
        # Email already exists
        return None
    finally:
        conn.close()


def authenticate_user(email, password):
    """Authenticate a user by email and password"""
    conn = sqlite3.connect("instance/roadmaps.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Hash the provided password
    hashed_password = hash_password(password)

    # Find the user by email and password
    cursor.execute(
        "SELECT id, name, email FROM users WHERE email = ? AND password = ?",
        (email, hashed_password),
    )

    user = cursor.fetchone()
    conn.close()

    return dict(user) if user else None


# Add to the existing init_db function or call separately
def initialize_databases():
    """Initialize all database tables"""
    try:
        # Make sure the instance directory exists
        if not os.path.exists("instance"):
            os.makedirs("instance")

        init_db()  # Initialize roadmaps table
        init_user_db()  # Initialize users table
        print("Databases initialized successfully")
    except Exception as e:
        print(f"Error initializing databases: {str(e)}")


def login_required(route_function):
    """Decorator to require login for certain routes"""

    @functools.wraps(route_function)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            # User is not logged in, redirect to login page
            return redirect(url_for("login", next=request.url))
        return route_function(*args, **kwargs)

    return decorated_function


app.jinja_env.filters["md_to_html"] = md_to_html


# Add these routes to handle login and registration
@app.route("/login", methods=["GET", "POST"])
def login():
    print(f"Login route called with method: {request.method}")
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            return render_template(
                "login.html", error="Please provide both email and password"
            )

        user = authenticate_user(email, password)

        if user:
            # Set user session
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["user_email"] = user["email"]

            # Redirect to dashboard or home page
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid email or password")

    # GET request - show the login form
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    print(f"Register route called with method: {request.method}")
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")

        if not name or not email or not password:
            return render_template("login.html", reg_error="Please fill all fields")

        # Register the user
        user_id = register_user(name, email, password)

        if user_id:
            # Set user session
            session["user_id"] = user_id
            session["user_name"] = name
            session["user_email"] = email

            # Redirect to dashboard
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", reg_error="Email already exists")

    # GET request - redirect to login page which has registration form
    return redirect(url_for("login"))


@app.route("/logout")
def logout():
    # Clear the user session
    session.clear()
    return redirect(url_for("login"))


# Route for the results page
@app.route("/results", methods=["GET", "POST"])
def results():
    roadmap = None
    error = None
    subject = None

    if request.method == "POST":
        # Get the subject from the form
        text_content = request.form.get("text_content")
        subject = text_content  # Store for displaying back to the user

        if text_content:
            try:
                # Generate the roadmap
                roadmap_data = generate_roadmap(text_content)

                if isinstance(roadmap_data, str) and roadmap_data.startswith("Error"):
                    error = roadmap_data
                    roadmap = generate_fallback_roadmap(text_content)
                else:
                    # Enhance roadmap with videos and pass the subject
                    _, roadmap = enhance_roadmap_with_videos(
                        roadmap_data, subject=text_content
                    )
            except Exception as e:
                error = f"Application error: {str(e)}"
                roadmap = generate_fallback_roadmap(text_content)
        else:
            error = "Please enter a subject for the roadmap"

    # Handle GET request with query parameter
    elif request.method == "GET" and request.args.get("query"):
        query = request.args.get("query")
        subject = query

        try:
            # Generate the roadmap
            roadmap_data = generate_roadmap(query)

            if isinstance(roadmap_data, str) and roadmap_data.startswith("Error"):
                error = roadmap_data
                roadmap = generate_fallback_roadmap(query)
            else:
                # Enhance roadmap with videos and pass the subject
                _, roadmap = enhance_roadmap_with_videos(roadmap_data, subject=query)
        except Exception as e:
            error = f"Application error: {str(e)}"
            roadmap = generate_fallback_roadmap(query)

    return render_template("index.html", roadmap=roadmap, error=error, subject=subject)


# Main route for the landing page
@app.route("/", methods=["GET"])
def main():
    # Get user info from session if logged in
    user = None
    if "user_id" in session:
        user = {
            "id": session.get("user_id"),
            "name": session.get("user_name"),
            "email": session.get("user_email"),
        }

    return render_template("index.html", user=user)


@app.route("/root")
def root():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    else:
        return render_template("login.html")


def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    else:
        return redirect(url_for("login.html"))


@app.route("/login-page", methods=["GET"])
def login_page():
    return render_template("login.html")


@app.route("/main")
def main_page():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    else:
        return redirect(url_for("login"))


# API endpoint to handle the search
@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Redirect to results page with the query
    return jsonify({"redirect": f"/results?query={query}"})


# Route to handle the form submission
@app.route("/generate", methods=["POST"])
def generate_from_main():
    query = request.form.get("promptInput")  # Changed from text_content to promptInput
    is_regeneration = request.form.get("regenerate") == "true"

    if not query:
        return redirect("/")

    try:
        # Generate the roadmap
        roadmap_data = generate_roadmap(query)

        if isinstance(roadmap_data, str) and roadmap_data.startswith("Error"):
            error = roadmap_data
            roadmap = generate_fallback_roadmap(query)
        else:
            # Enhance roadmap with videos and pass the subject
            _, roadmap = enhance_roadmap_with_videos(roadmap_data, subject=query)
            error = None
    except Exception as e:
        error = f"Application error: {str(e)}"
        roadmap = generate_fallback_roadmap(query)

    return render_template("generate.html", roadmap=roadmap, error=error, subject=query)


# API endpoint for roadmap JSON with videos
@app.route("/api/roadmap", methods=["POST"])
def api_roadmap():
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Generate the roadmap
        roadmap_data = generate_roadmap(query)

        if isinstance(roadmap_data, str) and roadmap_data.startswith("Error"):
            # Return a basic structure instead of an error
            fallback_data = [
                {
                    "month": "Month 1",
                    "milestone": "Getting Started",
                    "tasks": [
                        {
                            "task": f"Learn the fundamentals of {query}",
                            "estimated_time": "10 hours",
                            "resources": "Online tutorials and documentation",
                        }
                    ],
                }
            ]
            return jsonify({"roadmap": fallback_data, "warning": roadmap_data})

        # Enhance roadmap with videos
        enhanced_roadmap, _ = enhance_roadmap_with_videos(roadmap_data)

        return jsonify({"roadmap": enhanced_roadmap})
    except Exception as e:
        return jsonify({"error": f"Application error: {str(e)}"}), 500


# Add a health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@app.route("/save-roadmap", methods=["POST"])
@login_required
def save_roadmap():
    subject = request.form.get("subject")
    roadmap = request.form.get("roadmap")

    if not subject or not roadmap:
        return redirect(url_for("results"))

    # Store the roadmap ID in session for redirect purposes
    roadmap_id = save_roadmap_to_db(subject, roadmap, session.get("user_id"))
    session["current_roadmap_id"] = roadmap_id

    # Redirect to the dashboard
    return redirect(url_for("dashboard"))


def save_roadmap_to_db(subject, content, user_id="anonymous"):
    """Save roadmap to database and return the ID"""
    conn = sqlite3.connect("instance/roadmaps.db")
    cursor = conn.cursor()

    # Insert the roadmap into the database
    cursor.execute(
        "INSERT INTO roadmaps (subject, content, user_id) VALUES (?, ?, ?)",
        (subject, content, user_id),
    )

    # Get the ID of the inserted row
    roadmap_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return roadmap_id


def get_all_roadmaps_for_user(user_id):
    """Fetch all roadmaps for a user"""
    conn = sqlite3.connect("instance/roadmaps.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, subject, content, created_at FROM roadmaps WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    )

    roadmaps = cursor.fetchall()
    conn.close()

    # Convert to list of dictionaries for template use
    result = []
    for roadmap in roadmaps:
        result.append(
            {
                "id": roadmap["id"],
                "subject": roadmap["subject"],
                "content": roadmap["content"],
                "created_at": roadmap["created_at"],
            }
        )

    return result


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Dashboard view showing the user's saved roadmaps"""
    # Check if user is logged in
    if "user_id" not in session:
        return redirect(url_for("login"))

    # Get user_id and user info from session
    user_id = session.get("user_id")
    user = {
        "id": user_id,
        "name": session.get("user_name"),
        "email": session.get("user_email"),
    }

    # Get all roadmaps for the user
    roadmaps = get_all_roadmaps_for_user(user_id)

    # Get the current roadmap ID from session (if exists)
    current_roadmap_id = session.get("current_roadmap_id")

    # If there's a specific roadmap to display
    current_roadmap = None
    if current_roadmap_id:
        # Get the specific roadmap
        conn = sqlite3.connect("instance/roadmaps.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, subject, content, created_at FROM roadmaps WHERE id = ?",
            (current_roadmap_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            current_roadmap = {
                "id": row["id"],
                "subject": row["subject"],
                "content": row["content"],
                "created_at": row["created_at"],
            }

    # Render the dashboard template with user info
    return render_template(
        "main.html", roadmaps=roadmaps, roadmap=current_roadmap, user=user
    )


@app.route("/quiz")
def quiz_page():
    """Render the quiz page"""
    current_roadmap_id = session.get("current_roadmap_id")

    # If there's a specific roadmap to display
    subject = None
    if current_roadmap_id:
        # Get the specific roadmap
        conn = sqlite3.connect("instance/roadmaps.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, subject, content, created_at FROM roadmaps WHERE id = ?",
            (current_roadmap_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            subject = row["subject"]
        else:
            # If roadmap doesn't exist, redirect to dashboard
            return redirect(url_for("dashboard"))
    print(f"Quiz page called with subject: {subject}")
    return render_template("quiz.html", subject=subject)


@app.route("/view-roadmap/<int:roadmap_id>", methods=["GET"])
def view_roadmap(roadmap_id):
    """View a specific roadmap"""
    # Get user info from session
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    user = {
        "id": user_id,
        "name": session.get("user_name"),
        "email": session.get("user_email"),
    }

    # Get all roadmaps for the user
    roadmaps = get_all_roadmaps_for_user(user_id)

    # Get the specific roadmap
    conn = sqlite3.connect("instance/roadmaps.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, subject, content, created_at FROM roadmaps WHERE id = ?",
        (roadmap_id,),
    )

    row = cursor.fetchone()
    conn.close()

    print({row["subject"]})

    # If roadmap exists, render it
    if row:
        current_roadmap = {
            "id": row["id"],
            "subject": row["subject"],
            "content": row["content"],
            "created_at": row["created_at"],
        }
        # Store the current roadmap ID in the session
        session["current_roadmap_id"] = roadmap_id
        return render_template(
            "main.html", roadmaps=roadmaps, roadmap=current_roadmap, user=user
        )
    else:
        # If roadmap doesn't exist, redirect to dashboard
        return redirect(url_for("dashboard"))


# Get the local IP address
def get_local_ip():
    try:
        # Create a socket connection to an external server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google's DNS server
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"  # Fallback to localhost


@app.route("/generate-quiz", methods=["POST"])
def generate_quiz_route():
    """Handle quiz generation requests"""
    current_roadmap_id = session.get("current_roadmap_id")

    # If there's a specific roadmap to display
    subject = None
    if current_roadmap_id:
        # Get the specific roadmap
        conn = sqlite3.connect("instance/roadmaps.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, subject, content, created_at FROM roadmaps WHERE id = ?",
            (current_roadmap_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            subject = row["subject"]
    print(f"Quiz generation called with subject: {subject}")
    print()
    try:
        # Generate the quiz using the existing function
        quiz_data = generate_quiz(subject)

        # Ensure we have a valid response structure
        if not quiz_data or "mcqs" not in quiz_data or not quiz_data["mcqs"]:
            # Create a fallback quiz if the API returned empty or invalid data
            quiz_data = {
                "mcqs": [
                    {
                        "mcq": f"Sample question about {subject}?",
                        "options": {
                            "a": "First option",
                            "b": "Second option",
                            "c": "Third option",
                            "d": "Fourth option",
                        },
                        "correct": "a",
                        "topic": subject,
                        "difficulty": "Easy",
                        "explanation": "This is a sample explanation for the correct answer.",
                    }
                ]
            }
            return jsonify({"quiz": quiz_data, "warning": "Generated fallback quiz"})

        return jsonify({"quiz": quiz_data})
    except Exception as e:
        print(f"Error in quiz generation: {str(e)}")
        return jsonify({"error": f"Error generating quiz: {str(e)}"}), 500


@app.route("/take-quiz/<subject>")
def take_quiz(subject):
    current_roadmap_id = session.get("current_roadmap_id")

    # If there's a specific roadmap to display
    subject = None
    if current_roadmap_id:
        # Get the specific roadmap
        conn = sqlite3.connect("instance/roadmaps.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, subject, content, created_at FROM roadmaps WHERE id = ?",
            (current_roadmap_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            subject = row["subject"]
    """Render the quiz page for a specific subject"""
    return render_template("quiz.html", subject=subject)


# Run the Flask app
if __name__ == "__main__":
    # Initialize all databases first
    initialize_databases()

    # Move the templates to the right location
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if templates directory exists, create if not
    templates_dir = os.path.join(current_dir, "templates")
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)

    # Create static directory for CSS, JS, and images
    static_dir = os.path.join(current_dir, "static")
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Create subdirectories in static
    for subdir in ["css", "js", "Img"]:
        subdir_path = os.path.join(static_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)

    main_source = os.path.join(current_dir, "main.html")
    main_dest = os.path.join(templates_dir, "main.html")
    if os.path.exists(main_source) and not os.path.exists(main_dest):
        import shutil

        shutil.copy(main_source, main_dest)

    init_db()

    # Get the local IP address
    local_ip = get_local_ip()
    port = 5000

    print(f"Starting server...")
    print(f"Access the application on this device at: http://127.0.0.1:{port}")
    print(f"Access from other devices on your network at: http://{local_ip}:{port}")

    # Run the Flask app on all network interfaces (0.0.0.0)
    app.run(host="0.0.0.0", port=port, debug=True)
