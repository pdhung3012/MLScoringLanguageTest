const recast = require('recast');
const { Parser } = require("acorn");
var fs = require('fs');
fileName='/Users/hungphan/Downloads/hw2Folder/abkesjacob/abkesjacobhw2/validation1.js'
fileTextOutput='/Users/hungphan/Downloads/hw2Folder/abkesjacob/abkesjacobhw2/textJs.txt'

function getLeafNodes(leafNodes, obj,indent){
    console.log('visit node '+indent+' '+obj.type)
    if(obj.body){
        console.log('visit node '+indent+' '+obj.body.type+' '+obj.body.length)
    }
    if(obj.body && obj.body.length>0){
        indent=indent+1
        obj.body.forEach(function(child){getLeafNodes(leafNodes,child,indent)});
    } else if(obj.body){
        indent=indent+1
        getLeafNodes(leafNodes,obj.body,indent);
    }

    else{
//        console.log('print leaf '+obj.type)
        leafNodes.push(obj.type);
    }
}


const ast = Parser.parse(fs.readFileSync(fileName).toString());

//const body = acorn.parse(buffer).body
console.log(ast)

list = []
indent=0
getLeafNodes(list,ast,indent)
let textJs=list.join(' ')
/*
for (var i = 0; i < list.length; i++) {
    console.log(i+'\t'+list[i])
}
*/

fs.writeFile(fileTextOutput, textJs, (err) => {
    // throws an error, you could also catch it here
    if (err) throw err;

    // success case, the file was saved
    console.log('TextJS saved!');
});