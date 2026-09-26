"""Three small upstream MCP servers used by the demo, compose stack and tests.

They stand in for real third-party systems (a document store, a payments
provider, a ticketing system). Each one requires the credential the gateway
injects, so the tests can prove that injection happens and passthrough does not.
"""
