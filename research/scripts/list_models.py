import tinker


def main() -> None:
    client = tinker.ServiceClient()
    caps = client.get_server_capabilities()
    for m in caps.supported_models:
        print(m.model_name)


if __name__ == "__main__":
    main()
