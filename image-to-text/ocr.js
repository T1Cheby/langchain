const T = require("tesseract.js");
T.recognize("./record.png", "eng", {logger: e => console.log(e)})
.then(out=>console.log(out))