"""
Skrypt do generowania plików licencji dla bibliotek zewnętrznych.

Program odczytuje pliki THIRD_PARTY_LICENSES_backend.md i THIRD_PARTY_LICENSES_frontend.md
w folderze /backend, scala dane bibliotek, aktualizuje datę weryfikacji licencji,
generuje nowe pliki licencji oraz zaktualizowany plik THIRD_PARTY_LICENSES.md
w folderze /licences.
"""

import os
from pathlib import Path
import datetime
import re

# Data sprawdzenia licencji
CHECK_DATE = "2025-05-15"

# Poprawki dla licencji UNKNOWN i innych specjalnych przypadków
LICENSE_CORRECTIONS = {
    "pillow": "HPND License",
    "typing_extensions": "Python Software Foundation License",
    "UNKNOWN": "UNKNOWN",  # Zachowujemy UNKNOWN, ale korygujemy później w kodzie
    "Other/Proprietary License": "NVIDIA Proprietary License",
    "NVIDIA Proprietary Software": "NVIDIA Proprietary License",
    "BSD": "BSD License"  # Korekta dla torchvision
}

# Szablony pełnych tekstów licencji
LICENSE_TEXTS = {
    "MIT License": """MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.""",

    "BSD License": """BSD 3-Clause License

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.""",

    "Apache Software License": """Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

[Full text of the Apache License can be found at: http://www.apache.org/licenses/LICENSE-2.0]

Note: Due to the length of the Apache License, the full text is not included here. Please refer to the official Apache website for the complete license terms.""",

    "HPND License": """Historical Permission Notice and Disclaimer (HPND)

Permission to use, copy, modify, and distribute this software and its
documentation for any purpose and without fee is hereby granted, provided
that the above copyright notice appear in all copies and that both that
copyright notice and this permission notice appear in supporting
documentation, and that the name of the copyright holder not be used in
advertising or publicity pertaining to distribution of the software without
specific, written prior permission.

THE COPYRIGHT HOLDER DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO
EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY SPECIAL, INDIRECT OR
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.""",

    "Python Software Foundation License": """Python Software Foundation License
Version 2

1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"), and
   the Individual or Organization ("Licensee") accessing and otherwise using this
   software in source or binary form and its associated documentation.

2. Subject to the terms and conditions of this License Agreement, PSF hereby
   grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
   analyze, test, perform and/or display publicly, prepare derivative works,
   distribute, and otherwise use this software alone or in any derivative
   version, provided, however, that PSF's License Agreement and PSF's notice of
   copyright, i.e., "Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007,
   2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
   2020, 2021, 2022, 2023 Python Software Foundation; All Rights Reserved" are
   retained in this software alone or in any derivative version prepared by
   Licensee.

[Full text of the license can be found at: https://docs.python.org/3/license.html]

Note: The full PSF License is extensive and includes additional terms. Please refer to the official Python documentation for the complete license text.""",

    "NVIDIA Proprietary License": """NVIDIA Proprietary License

This software is licensed under the NVIDIA Software License Agreement.
The full text of the license can be found at:
https://www.nvidia.com/en-us/drivers/nvidia-license/

Please refer to the NVIDIA website for the specific terms and conditions. The license may include restrictions on redistribution and use, which must be reviewed before using the software in commercial applications.""",

    "GNU General Public License (GPL)": """GNU General Public License
Version 2, June 1991

[Full text of the GPL can be found at: https://www.gnu.org/licenses/gpl-2.0.txt]

Note: The GNU General Public License is a copyleft license that requires derivative works to be distributed under the same license. The full text is not included here due to its length. Please review the official GPL text for complete terms and conditions.""",

    "The Unlicense (Unlicense)": """The Unlicense

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>""",

    "FreeBSD": """FreeBSD License

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.""",

    "MIT License; Mozilla Public License 2.0 (MPL 2.0)": """MIT License; Mozilla Public License 2.0 (MPL 2.0)

This software is dual-licensed under the MIT License and the Mozilla Public License 2.0.

### MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Mozilla Public License 2.0
[Full text of the MPL 2.0 can be found at: https://www.mozilla.org/en-US/MPL/2.0/]

Note: The MPL 2.0 is a more complex license that allows for both open-source and proprietary use under certain conditions. Please review the full text for complete terms.""",

    "MIT AND Python-2.0": """MIT License; Python-2.0 License

This software is dual-licensed under the MIT License and the Python-2.0 License.

### MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Python-2.0 License
[Full text of the Python-2.0 License can be found at: https://docs.python.org/3/license.html]

Note: The Python-2.0 License is part of the Python Software Foundation License. Please refer to the official Python documentation for the complete license text.""",

    "Apache Software License; BSD License": """Apache Software License; BSD License

This software is dual-licensed under the Apache Software License and the BSD License.

### Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

[Full text of the Apache License can be found at: http://www.apache.org/licenses/LICENSE-2.0]

Note: Due to the length of the Apache License, the full text is not included here. Please refer to the official Apache website for the complete license terms.

### BSD License
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.""",

    "Apache Software License; MIT License": """Apache Software License; MIT License

This software is dual-licensed under the Apache Software License and the MIT License.

### Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

[Full text of the Apache License can be found at: http://www.apache.org/licenses/LICENSE-2.0]

Note: Due to the length of the Apache License, the full text is not included here. Please refer to the official Apache website for the complete license terms.

### MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
}

def parse_old_licenses_file(file_path):
    """
    Parsuje plik THIRD_PARTY_LICENSES_*.md i zwraca listę bibliotek.
    Obsługuje różne kodowania (UTF-8, UTF-16, ISO-8859-1).
    
    Args:
        file_path (Path): Ścieżka do pliku THIRD_PARTY_LICENSES_*.md
        
    Returns:
        list: Lista słowników zawierających dane bibliotek
    """
    libraries = []
    encodings = ['utf-8', 'utf-16', 'iso-8859-1']  # Lista próbnych kodowań
    
    content = None
    used_encoding = None
    
    # Próbuj otworzyć plik z różnymi kodowaniami
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                used_encoding = encoding
                break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        print(f"Błąd: Nie udało się odczytać pliku {file_path} w żadnym z kodowań: {', '.join(encodings)}")
        return []
    
    print(f"Wczytano plik {file_path} z kodowaniem {used_encoding}")
    
    try:
        # Usuń znacznik BOM, jeśli istnieje
        content = content.lstrip('\ufeff')
        
        # Usuń komentarze HTML (<!-- -->), które mogą przeszkadzać
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        
        # Wyszukujemy wiersze tabeli (3 kolumny: Name, Version, License)
        row_pattern = r"\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|"
        rows = re.findall(row_pattern, content)
        
        for row in rows:
            # Pomijamy wiersz nagłówka i separator
            if row[0].strip() == "Name" or row[0].strip().startswith("-"):
                continue
                
            libraries.append({
                "name": row[0].strip(),
                "version": row[1].strip(),
                "license": row[2].strip()
            })
        
        if libraries:
            print(f"Wczytano {len(libraries)} bibliotek z pliku {file_path}")
        else:
            print(f"Nie znaleziono bibliotek w pliku {file_path}. Sprawdź format tabeli.")
            
        return libraries
    except Exception as e:
        print(f"Błąd podczas parsowania pliku {file_path}: {e}")
        return []

def merge_libraries(backend_libs, frontend_libs):
    """
    Scala biblioteki z backendu i frontendu, eliminując duplikaty.
    
    Args:
        backend_libs (list): Lista bibliotek z backendu
        frontend_libs (list): Lista bibliotek z frontendu
        
    Returns:
        list: Połączona lista bibliotek z adnotacją źródła
    """
    merged_libs = {}
    
    # Przetwarzaj biblioteki z backendu
    for lib in backend_libs:
        key = (lib['name'], lib['version'])
        merged_libs[key] = {
            'name': lib['name'],
            'version': lib['version'],
            'license': lib['license'],
            'source': ['Backend']
        }
    
    # Przetwarzaj biblioteki z frontendu
    for lib in frontend_libs:
        key = (lib['name'], lib['version'])
        if key in merged_libs:
            # Biblioteka już istnieje, dodaj źródło
            merged_libs[key]['source'].append('Frontend')
            # Sprawdź, czy licencje się zgadzają
            if merged_libs[key]['license'] != lib['license']:
                print(f"Ostrzeżenie: Biblioteka {lib['name']} ({lib['version']}) ma różne licencje: "
                      f"Backend: {merged_libs[key]['license']}, Frontend: {lib['license']}. "
                      f"Zachowuję licencję z backendu.")
        else:
            # Nowa biblioteka
            merged_libs[key] = {
                'name': lib['name'],
                'version': lib['version'],
                'license': lib['license'],
                'source': ['Frontend']
            }
    
    # Konwertuj na listę
    return [lib for lib in merged_libs.values()]

def generate_license_file(library, licences_dir):
    """Generuje plik *_LICENSE.md dla danej biblioteki."""
    name = library['name']
    version = library['version']
    license_name = library['license']
    check_date = library.get('check_date', CHECK_DATE)
    source = ', '.join(library['source'])

    # Nazwa pliku licencji
    safe_name = name.replace('-', '_').replace(' ', '_')
    license_file = f"{safe_name}_LICENSE.md"
    license_path = licences_dir / license_file

    # Treść licencji
    license_text = LICENSE_TEXTS.get(license_name, f"The full text of the {license_name} can be found at the official project repository or website for {name}.")
    
    # Szablon pliku licencji
    content = f"""# License for {name}

## Library Information
- **Name**: {name}
- **Version**: {version}
- **License**: {license_name}
- **License Check Date**: {check_date}
- **Source**: {source}
- **Project Source**: https://pypi.org/project/{name}/

## License Text
{license_text}
"""
    
    with open(license_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return license_file

def generate_third_party_licenses(libraries, licences_dir):
    """Generuje plik THIRD_PARTY_LICENSES.md."""
    content = f"""# Third-Party Licenses

This project uses the following third-party libraries. The licenses were verified on **{CHECK_DATE}**. Full license texts are available in the `licences/` directory.

| Name | Version | License | License Check Date | Source | License File |
|------|---------|---------|--------------------|--------|--------------|
"""
    
    for lib in sorted(libraries, key=lambda x: x['name'].lower()):
        lib['check_date'] = CHECK_DATE
        license_file = generate_license_file(lib, licences_dir)
        source = ', '.join(lib['source'])
        content += f"| {lib['name']} | {lib['version']} | {lib['license']} | {lib['check_date']} | {source} | [{license_file}]({license_file}) |\n"
    
    third_party_path = licences_dir / "THIRD_PARTY_LICENSES.md"
    with open(third_party_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    # Ścieżki
    inne_dir = Path("backend/Inne")
    licences_dir = Path("licences")
    backend_licenses_path = inne_dir / "THIRD_PARTY_LICENSES_backend.md"
    frontend_licenses_path = inne_dir / "THIRD_PARTY_LICENSES_frontend.md"
    
    # Sprawdź, czy pliki licencji istnieją
    if not backend_licenses_path.exists():
        print(f"Plik {backend_licenses_path} nie istnieje!")
        return
    if not frontend_licenses_path.exists():
        print(f"Plik {frontend_licenses_path} nie istnieje!")
        return
    
    # Stwórz folder licences jeśli nie istnieje
    licences_dir.mkdir(exist_ok=True)
    
    # Wczytaj dane z plików licencji
    backend_libs = parse_old_licenses_file(backend_licenses_path)
    frontend_libs = parse_old_licenses_file(frontend_licenses_path)
    
    if not backend_libs and not frontend_libs:
        print("Nie udało się wczytać danych bibliotek z żadnego pliku.")
        print("Brak bibliotek do przetworzenia. Program zostanie zakończony.")
        return
    
    # Scal biblioteki
    libraries = merge_libraries(backend_libs, frontend_libs)
    
    # Popraw specjalne przypadki licencji
    for lib in libraries:
        # Popraw według tabeli korekcji
        current_license = lib['license']
        if current_license in LICENSE_CORRECTIONS:
            lib['license'] = LICENSE_CORRECTIONS[current_license]
        
        # Dodatkowa korekta dla konkretnych pakietów NVIDIA
        if lib['name'].startswith('nvidia-') and "Proprietary" not in lib['license']:
            lib['license'] = "NVIDIA Proprietary License"
        
        # Specjalna korekta dla UNKNOWN
        if lib['license'] == "UNKNOWN" and lib['name'] in LICENSE_CORRECTIONS:
            lib['license'] = LICENSE_CORRECTIONS[lib['name']]
    
    # Wygeneruj THIRD_PARTY_LICENSES.md i pliki licencji
    generate_third_party_licenses(libraries, licences_dir)
    
    print(f"Wygenerowano THIRD_PARTY_LICENSES.md i pliki licencji w folderze {licences_dir}")

if __name__ == "__main__":
    main()