from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo 🚀")


@mcp.tool()
def get_weather(location: str) -> str:
    return f"The weather in {location} is a perfect sunny 70°F today. Enjoy your day!"

@mcp.tool()
def get_time(location: str) -> str:
    return f"The current time in {location} is 3:00 PM. Have a great afternoon!"

if __name__ == "__main__":
    mcp.run(transport="sse")
