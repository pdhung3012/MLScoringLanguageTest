

function getLeafNodes(leafNodes, obj,indent){
    //console.log('visit node '+indent+' '+obj.type)
    if(obj.body){
       // console.log('visit node '+indent+' '+obj.body.type+' '+obj.body.length)
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

function getSubFolder(fopInput){
    return fs.readdirSync(fopInput).filter(function (file) {
        return fs.statSync(fopInput+'/'+file).isDirectory();
    });
    return results;
}

var strResult=''

function readTextFile(fopInput){
//    strResult='';
    list=[]
    fs.readFileSync(fopInput, 'utf8', function(err, data) {
        if (err) console.log(err);
        strResult=data;
//        console.log('data here'+data);
    });

    return strResult;
}

const recast = require('recast');
const { Parser } = require("acorn");
var fs = require('fs');

folderInput='/Users/hungphan/Downloads/hw2TextualTree/'
fileName='/Users/hungphan/Downloads/hw2Folder/abkesjacob/abkesjacobhw2/validation1.js'
fileTextOutput='/Users/hungphan/Downloads/hw2Folder/abkesjacob/abkesjacobhw2/textJs.txt'
fnJSList='listJSFiles.txt'
fnTextJS='textJs.txt'

listFolders=getSubFolder(folderInput)
for (var i = 0; i < listFolders.length; i++) {
    var fileListJSNames=folderInput+'/'+listFolders[i]+"/listJSFiles.txt"
    console.log(fileListJSNames)
   // strList=readTextFile(fileListJSNames);
    strList= fs.readFileSync(fileListJSNames, 'utf8');
    /*fs.readFileSync(fileListJSNames, 'utf8', function(err, data) {
        if (err) console.log(err);
        strList=data;
//        console.log('data here'+data);
    });*/

    console.log('shot '+strList);
    var listJSFiles=strList.trim().split('\n');
    list = []
    console.log(listJSFiles);

    for(var j = 0; j <listJSFiles.length; j++){
        indent = 0;
        console.log(j+'\t'+listJSFiles[j]);
        try {
            ast = Parser.parse(fs.readFileSync(listJSFiles[j]).toString());
            getLeafNodes(list,ast,indent);
        }catch(e){
            console.log(e)
        }

    }

    var textJs=list.join(' ')
    fileTextOutput=folderInput+'/'+listFolders[i]+"/textJs.txt"
    fs.writeFile(fileTextOutput, textJs, (err) => {
        console.log('TextJS saved!');
    });


    console.log(i+'\t'+listFolders[i]);
}

/*
const ast = Parser.parse(fs.readFileSync(fileName).toString());
console.log(ast)
list = []
indent=0
*/

/*
let textJs=list.join(' ')

fs.writeFile(fileTextOutput, textJs, (err) => {
    // throws an error, you could also catch it here
    if (err) throw err;

    // success case, the file was saved
    console.log('TextJS saved!');
});
*/