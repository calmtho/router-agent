import sys

errors = []
modules = [
    'app.services.typo_service',
    'app.config',
    'app.graph.state',
    'app.graph.nodes',
    'app.chains.rag_chain',
    'app.agents.chat_agent',
    'app.agents.rag_agent',
    'app.agents.mcp_agent',
    'app.routers.chat',
]
for m in modules:
    try:
        __import__(m)
        print(f'OK   {m}')
    except Exception as e:
        print(f'FAIL {m}: {e}')
        errors.append((m, e))

if errors:
    sys.exit(1)
print('\nAll imports OK!')
