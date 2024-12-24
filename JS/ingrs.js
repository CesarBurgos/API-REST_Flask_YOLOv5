const sub_arch = function(label, file){
    if(document.getElementById(file).files.length !== 0){
        document.getElementById(label).style.fontStyle = 'Italic';
        document.getElementById(label).innerText = document.getElementById(file).files[0].name;
    }else{
        document.getElementById(label).style.fontStyle = 'normal';
        document.getElementById(label).innerText = 'Hacer clic aquí para buscar su archivo';
    }
}