import logging
import sys


def configure_logging(service_name: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s {service_name} %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
