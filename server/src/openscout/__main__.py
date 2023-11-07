#
# OpenScout
#   - Distributed Automated Situational Awareness
#
#   Author: Thomas Eiszler <teiszler@andrew.cmu.edu>
#
#   Copyright (C) 2020 Carnegie Mellon University
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import argparse
import logging

from gabriel_server.network_engine import server_runner

FIVE_SECONDS = 5

DEFAULT_PORT = 9099
DEFAULT_NUM_TOKENS = 2
INPUT_QUEUE_MAXSIZE = 60
SOURCE = "openscout"

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-t", "--tokens", type=int, default=DEFAULT_NUM_TOKENS, help="number of tokens"
    )

    parser.add_argument(
        "-p", "--port", type=int, default=DEFAULT_PORT, help="Set port number"
    )

    parser.add_argument(
        "-q",
        "--queue",
        type=int,
        default=INPUT_QUEUE_MAXSIZE,
        help="Max input queue size",
    )
    
    
    parser.add_argument(
        "--timeout",
        default=FIVE_SECONDS,
        type = int,
        help="timeout period from server to client",
    )

    args, _ = parser.parse_known_args()

    server_runner.run(
        websocket_port=args.port,
        zmq_address="tcp://*:5555",
        num_tokens=args.tokens,
        input_queue_maxsize=args.queue,
        timeout = args.timeout
    )


if __name__ == "__main__":
    main()
