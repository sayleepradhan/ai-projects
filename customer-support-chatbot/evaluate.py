"""evaluate.py — Quick evaluation of RAG pipeline quality."""

from chain import CustomerSupportChain

chain = CustomerSupportChain()

# Test queries covering different intents
test_cases = [
    {
        "query": "How do I cancel my order?",
        "expected_intent": "cancel_order",
    },
    {
        "query": "What payment methods do you accept?",
        "expected_intent": "check_payment_methods",
    },
    {
        "query": "I want to track my refund",
        "expected_intent": "track_refund",
    },
    {
        "query": "How do I change my shipping address?",
        "expected_intent": "change_shipping_address",
    },
    {
        "query": "I forgot my password",
        "expected_intent": "recover_password",
    },
    {
        "query": "What is your return policy?",
        "expected_intent": "check_refund_policy",
    },
    # Edge case: question outside the knowledge base
    {
        "query": "What is the meaning of life?",
        "expected_intent": None,  # Should acknowledge it can't help
    },
]

print("RAG Pipeline Evaluation")
print("=" * 70)

correct_retrievals = 0
total_with_expected = 0

for tc in test_cases:
    result = chain.ask(tc["query"])
    retrieved_intents = [c["intent"] for c in result["sources"]]

    # Check if the expected intent appears in retrieved sources
    expected = tc["expected_intent"]
    if expected:
        total_with_expected += 1
        hit = expected in retrieved_intents
        if hit:
            correct_retrievals += 1
        status = "HIT" if hit else "MISS"
    else:
        status = "N/A (out-of-scope)"

    print(f"\nQ: {tc['query']}")
    print(f"Expected intent: {expected}")
    print(f"Retrieved intents: {retrieved_intents}")
    print(f"Retrieval: {status}")
    print(f"Response: {result['response'][:200]}...")

if total_with_expected > 0:
    accuracy = correct_retrievals / total_with_expected * 100
    print(f"\n{'='*70}")
    print(f"Retrieval accuracy: {correct_retrievals}/{total_with_expected} ({accuracy:.0f}%)")