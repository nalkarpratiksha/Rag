from utils.metadata_search import rag_metadata_qa
from utils.hybrid_search import hybrid_rag_qa


def main():
    question = input("Enter your question: ").strip()

    print(
        "\nChoose retrieval method:\n"
        "1️ Metadata Search\n"
        "2️ Hybrid Search"
    )

    try:
        choice = int(input("Enter choice (1 or 2): ").strip())
    except ValueError:
        print(" Invalid input. Please enter 1 or 2.")
        return

    if choice == 1:
        answer = rag_metadata_qa(
            question=question,
            metadata_key="page",
            metadata_value=2,
            top_k=3
        )

    elif choice == 2:
        answer = hybrid_rag_qa(
            question=question,
            top_k=3
        )

    else:
        print(" Invalid choice!")
        return

    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
