#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
try:
    import cElementTree as ET
except ImportError:
    try:
        # Python 2.5 need to import a different module
        import xml.etree.cElementTree as ET
    except ImportError:
        exit_err("Failed to import cElementTree from any known place")

CONFIG_XML_DATA = """
<config name='dClaw1 dClaw2'>
  <limits low="1 2" high="2 3"/>
  <scale joint="10 20"/>
  <data type="test1 test2"/>
</config>
"""


# Read config from root
def read_config_from_node(root_node, parent_name, child_name, dtype=int):
    # find parent
    parent_node = root_node.find(parent_name)
    if parent_node == None:
        quit("Parent %s not found" % parent_name)

    # get child data
    child_data = parent_node.get(child_name)
    if child_data == None:
        quit("Child %s not found" % child_name)

    config_val = np.array(child_data.split(), dtype=dtype)
    return config_val


# get config frlom file or string
def get_config_root_node(config_file_name=None, config_file_data=None):
    try:
        # get root
        if config_file_data is None:
            config_file_content = open(config_file_name, "r")
            config = ET.parse(config_file_content)
            root_node = config.getroot()
        else:
            root_node = ET.fromstring(config_file_data)

        # get root data
        root_data = root_node.get('name')
        root_name = np.array(root_data.split(), dtype=str)
    except:
        quit("ERROR: Unable to process config file %s" % config_file_name)

    return root_node, root_name


# Read config from config_file
def read_config_from_xml(config_file_name, parent_name, child_name, dtype=int):
    root_node, root_name = get_config_root_node(
        config_file_name=config_file_name)
    return read_config_from_node(root_node, parent_name, child_name, dtype)


# tests
if __name__ == '__main__':
    print("Read config and parse -------------------------")
    root, root_name = get_config_root_node(config_file_data=CONFIG_XML_DATA)
    print("Root:name \t", root_name)
    print("limit:low \t", read_config_from_node(root, "limits", "low", float))
    print("limit:high \t", read_config_from_node(root, "limits", "high", float))
    print("scale:joint \t", read_config_from_node(root, "scale", "joint",
                                                  float))
    print("data:type \t", read_config_from_node(root, "data", "type", str))

    # read straight from xml (dumb the XML data as duh.xml for this test)
    root, root_name = get_config_root_node(config_file_name="duh.xml")
    print("Read from xml --------------------------------")
    print("limit:low \t", read_config_from_xml("duh.xml", "limits", "low",
                                               float))
    print("limit:high \t",
          read_config_from_xml("duh.xml", "limits", "high", float))
    print("scale:joint \t",
          read_config_from_xml("duh.xml", "scale", "joint", float))
    print("data:type \t", read_config_from_xml("duh.xml", "data", "type", str))
