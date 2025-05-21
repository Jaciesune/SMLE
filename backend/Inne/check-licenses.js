const checker = require('license-checker');
const path = require('path');
const fs = require('fs');

//Aby uruchomić ten skrypt, użyj polecenia: node check-licenses.js
const appPath = path.resolve(__dirname, '../../SMLE app/SMLE');

checker.init({
  start: appPath,
  production: true,
  json: true
}, function (err, packages) {
  if (err) {
    console.error('Błąd podczas sprawdzania licencji:', err);
    return;
  }

  const output = {};
  for (const [pkgPath, info] of Object.entries(packages)) {
    const relativePath = path.relative(appPath, info.path || '');
    output[pkgPath] = {
      licenses: info.licenses,
      repository: info.repository,
      licenseFile: info.licenseFile ? path.relative(appPath, info.licenseFile) : null,
      relativePath: relativePath
    };
  }

  // Zapisz plik wynikowy obok skryptu (czyli w backend/Inne)
  const outputPath = path.resolve(__dirname, 'licenses.json');
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`✔ Licencje zapisane w ${outputPath}`);
});
