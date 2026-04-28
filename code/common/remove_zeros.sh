#!/bin/bash

path="/d/Stacked/Mapping_Population_2023/6-27-2023_5dpi/1/"
find "$path" -type f -name "*-Sym.png" -exec bash -c 'mv "$0" "${0/0*/}"' {} \;
