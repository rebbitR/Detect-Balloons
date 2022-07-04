
import logging

def write_to_log(txt):
    logging.info(txt)

def writeToFile(txt):
    f = open('./results_of_3models.txt', 'a')
    f.write(txt+"\n")
    f.close()

def writeToFileRes(type_model,kind):
    f = open('./results_of_3models.txt', 'a')
    f.write("-model: "+type_model+" -class: "+kind+"\n")
    f.close()



def getLog():
    with open('./text.txt', 'r') as f:
        f_contents = f.read()
        return f_contents




def CreateLog():
    logging.basicConfig(filename="log.txt", level=logging.DEBUG)


if __name__ == '__main__':
    CreateLog()