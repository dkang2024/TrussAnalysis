#Importing
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import os

#Functions

#Create the JSON File
def createjson():
    
    Full_Storage = {"Matrix_Storage": {}, "Materials_Matrix": {}}
    
    if not os.path.exists("MathModelStorage.json"):
        
        create = open("MathModelStorage.json", "x")
        
        with open("MathModelStorage.json", "w") as writing:
            json.dump(Full_Storage, writing)

#If an angle value is close, we round because of inaccuracy with Radians
def rounding(number):
    
    #If it's close within 10 decimal places to the rounded number, we round
    if math.isclose(number, round(number), abs_tol = 1e-10):
        return round(number)
    
    #If it's not, we just return
    else:
        return number

#Rounding for graphing
def roundingforgraphing(number):

    return round(number, 2)

#Function for finding the Node IDs on an element
def findnodes(elementid, E):
    
    return (E[int(elementid) - 1, 1], E[int(elementid) - 1, 2])

#Function for checking if the structure is two-dimensional
def checkif2d(N):
    
    nodezpos = []
    
    for noderow in N[:, 3]:
        nodezpos.append(noderow)
    
    if all(element == nodezpos[0] for element in nodezpos):
        return True
    
    return False

#Function for Finding Young's Modulus Multiplied by Cross-Sectional Area for the Specific Element
def findingcomponents(elementname, thingtofind):
    
    while True:
        
        try:
            returning = float(input(f"What's this element(element {elementname})'s {thingtofind}? "))
            return returning
        
        except:
            print(f"You didn't enter a valid number for the element's {thingtofind}!")

#Function for Establishing the Local Stiffness Matrix for the Specific Element
def stiffness(EA, L):

    return np.array([[EA/L, -EA/L], [-EA/L, EA/L]])

#Converting the Local Stiffness Matrix Into a Global Stiffness Matrix
def converttoglobalstiffness(Conversion_Matrix, stiffnessmatrix):

    return np.round(np.matmul(np.matmul(np.transpose(Conversion_Matrix), stiffnessmatrix), Conversion_Matrix), 3)

#Shaving Off Parts of the Global Stiffness Matrix
def removepartsglobalstiffness(globalmatrix, reactionlist):
    
    deletion_list = []
    
    for septuple in reactionlist:
        deletion_list.append(int(septuple[1]))
    
    newarray = np.delete(globalmatrix, deletion_list, 0)
    return np.delete(newarray, deletion_list, 1)

#Finding the Displacements Using the Shaved Global Stiffness Matrix
def finddisplacements(shavedstiffness, globalforce):
    return np.matmul(np.linalg.inv(shavedstiffness), globalforce)

#Solving For The Element's Material Tension vs Compression
def findelementforce(elementdisplacement, localstiffness, conversion):

    forces_list = np.matmul(np.matmul(localstiffness, conversion), elementdisplacement).tolist()
    final_forces_list = []

    for internal_list in forces_list:

        final_forces_list.append(rounding(internal_list[0]))
    
    return final_forces_list

#Find Reaction Forces
def findreactionforces(globalstiffness, rlist, globaldisplacement):
    
    full_matrix = np.matmul(globalstiffness, globaldisplacement)
    reactions = []
    
    for position in rlist[:, 1]:
        index = np.flatnonzero(position == rlist[:, 1])
        final_reaction = full_matrix[int(position), 0] - rlist[index, 2]
        reactions.append([float(rlist[index, 0]), rounding(float(final_reaction))])
    
    return np.array(reactions)

#Checking For Node Duplicates Function
def Check_Node_Duplicates(value_1, value_2, value_3, array, what_check, iter):
    
    for i in range(1, iter + 1):
        
        if [i, value_1, value_2, value_3] in array:
            
            print(f"This {what_check} is something that you've already entered! Deleting this {what_check}...")
            return False
    
    return True

#Checking Duplicates for Elements
def Check_Element_Duplicates(value_1, value_2, array, what_check, iter):
    
    for i in range(1, iter + 1):
        
        if [i, value_1, value_2] in array:
            
            print(f"This {what_check} is something that you've already entered! Deleting this {what_check}...")
            return False
    
    return True

#Checking Duplicates for Reaction Nodes
def Check_Reaction_Duplicates(node, array, what_check):
    
    if node in array[:][0]:
        
        print(f"This {what_check} is something that you've already entered! Deleting this {what_check}...")
        return False
    
    return True

#Initializing the Arrays Function
def Init_Array(Array_Name):
    
    array_input = False
    
    while not array_input:
        
        try:
            
            array_num = int(input(f"How many {Array_Name} do you have? "))
            array_input = True
        
        except:
            
            print("You didn't enter a valid number! Try again!")
    
    return array_num

#Function for Determining Which Directions A Node Is Free to Move
def DOF(direction):
    
    while True:
        
        spec_DOF = input(f"Is there a reaction force(aka is the node not able to move) in the {direction} direction? (y for not able to move/n for able to move) ")
        
        if spec_DOF.lower() == "y":
            
            return 1.0
        
        elif spec_DOF.lower() == "n":
            
            return 2.0
        
        else:
            
            print("You must enter y or n to indicate valuable information about the reaction force!")

#Storage For The Entire Matrix
def Matrix_Storage(matrix):
    
    matrix_storage = False
    
    while not matrix_storage:
        
        do_store = input("Do you want to store the entire matrix you just entered? (y/n) ")
        
        if do_store.lower() == "y":
            
            name = input("What do you want to name this stored matrix? ")
            
            with open("MathModelStorage.json", "r") as reading:
                
                temp = json.load(reading)
                temp["Matrix_Storage"][name] = matrix.tolist()
            
            with open("MathModelStorage.json", "w") as writing:
                
                json.dump(temp, writing)
            
            matrix_storage = True
        
        elif do_store.lower() == "n":
            
            matrix_storage = True
        
        else:
            
            print("Indiciate y/n so that the program knows your wishes!")

#Seeing if the User wants to pull out a Matrix
def Matrix_Pull(whatpull):
    
    while True:
        
        s_pull = input(f"Do you want to use a matrix of {whatpull} that you previously stored? (y/n) ")
        
        if s_pull.lower() == "y":
            
            while True:
                
                name = input("What's the name of said matrix that you inputted? ")
                
                try:
                    
                    with open("MathModelStorage.json", "r") as reading:
                        
                        temp = json.load(reading)
                        matrix = np.array(temp["Matrix_Storage"][name])
                        
                        return matrix, int(np.shape(matrix)[0]), True
                
                except:
                    
                    print("This name is invalid since it doesn't exist!")
        
        elif s_pull.lower() == "n":
            
            return 0, 0, False
        
        else:
            
            print("You didn't enter y/n to express your wishes!")

#Creating an Element Material Matrix
def Create_Element_Matrix(E):

    valid_create = False

    while not valid_create:

        store_materials, materials_list = input("Do you want to create an element material list based off the elements you have inputted currently? (y/n) "), []
        
        if store_materials.lower() == "y":

            for elementid in E[:, 0]:

                valid_material = False

                while not valid_material:

                    try:

                        material = float(input(f"What's this element's(element {str(elementid)}) Young's Modulus multiplied by Cross-Sectional Area? "))
                        materials_list.append(material)
                        valid_material = True
                    
                    except:

                        print("You didn't enter a valid number for the element's Young's Modulus multiplied by Cross-Sectional Area!")
            
            with open("MathModelStorage.json", "r") as reading:

                temp = json.load(reading)
            
            name = input("What do you want to name this stored material list? ")
            temp["Materials_Matrix"][name] = materials_list
            
            with open("MathModelStorage.json", "w") as writing:

                json.dump(temp, writing)
            
            valid_create = True
             
        elif store_materials.lower() == "n":

            print("Alright!")
            valid_create = True
        
        else:

            print("You must enter 'y' or 'n' for the program to recognize your wishes!")

#Utilizing an Element Material Matrix
def Utilize_Element_Matrix(E):

    while True:
        
        run_pull_matrix = input("Do you want to use an element material matrix that you stored? (y/n) ")

        if run_pull_matrix.lower() == "y":
             
            while True:

                name = input("What's the name of the element material list you inputted? ")
                    
                try:

                    with open("MathModelStorage.json", "r") as reading:

                        temp = json.load(reading)
                        
                    materials_list = temp["Materials_Matrix"][name]

                    if len(materials_list) == E.shape[0]:

                        return True, materials_list
                    
                    else:

                        print("This element material list is incompatible with your current elements! ")

                except:
                        
                    print("This element material list does not exist! Try again! ")
        
        elif run_pull_matrix.lower() == "n":

            return False, 0
        
        else:
            
            print("You must enter 'y' or 'n' in order to indicate your wishes for the program!")

#General Function For Finding Values for forming a matrix
def Find_Values(What_To_Find, Acting_Where, should_check, exist_check, v):
    while True:
                
        try:
            
            x = float(input(f"Enter your {What_To_Find}'s({What_To_Find} {str(v)}) {Acting_Where}: "))
            
            if should_check == True:
                
                in_array = np.flatnonzero(x == exist_check)
                
                if len(in_array) == 0:
                    print(f"You entered a {Acting_Where} that does not exist!")
                
                else:
                    return x
            
            else:
                return x
        
        except:
            print(f"You can only enter a valid number that represents your {Acting_Where}!")

def main():
    createjson()

    #Give Option to Pull Matrix for the Node Values
    function_receive = Matrix_Pull("node values")
    
    if function_receive[2] is True:
        
        N = function_receive[0]
        Nt = function_receive[1]
    
    #If the User doesn't want to, let the user create the Node Values
    else:
        
        Nt = Init_Array("nodes")
        N = []
        
        for i in range(0, Nt):
            
            v = i + 1
            no_overlap = False
            
            while not no_overlap:
                Node_x = Find_Values("node", "x-coordinate", False, 0, v)
                Node_y = Find_Values("node", "y-coordinate", False, 0, v)
                Node_z = Find_Values("node", "z-coordinate", False, 0, v)
                no_overlap = Check_Node_Duplicates(Node_x, Node_y, Node_z, N, "node", v)
            
            N.append([float(v), Node_x, Node_y, Node_z])
        N = np.array(N, dtype = "float")
        Matrix_Storage(N)
    
    print(N)   

    #Give Option to Pull Matrix for the Element Values
    function_receive = Matrix_Pull("element connections")

    if function_receive[2] is True:

        E = function_receive[0]
        Et = function_receive[1]
    
    #If the User doesn't want to, let the user create the Element Values
    else:
        
        Et = Init_Array("elements")
        E = []

        for i in range(0, Et):

            v = i + 1
            no_overlap = False

            while not no_overlap:
                EN1 = Find_Values("element", "node 1", True, N[:, 0], v)
                EN2 = Find_Values("element", "node 2", True, N[:, 0], v)

                if EN1 != EN2:
                    
                    if Check_Element_Duplicates(EN1, EN2, E, "element", v) is True:
                        no_overlap = True
                    
                else:
                    print("Your element cannot connect to the same node twice!")
            E.append([float(v), EN1, EN2])
        E = np.array(E, dtype = "float")
        Matrix_Storage(E)

    print(E)

    #Give Option to Pull Matrix for the External Force Values
    
    function_receive = Matrix_Pull("external force values")
    
    if function_receive[2] is True:
        
        F = function_receive[0]
        ExF = function_receive[1]

    #If the User doesn't want to, let the user create the External Force Values
    else:
        
        ExF = Init_Array("external forces")
        F = []
        
        for i in range(0, ExF):
            
            v = i + 1
            
            EFN = Find_Values("external force", "node", True, N[:, 0], v)
            EFX = Find_Values("external force", "x magnitude", False, 0, v)
            EFY = Find_Values("external force", "y magnitude", False, 0, v)
            EFZ = Find_Values("external force", "z magnitude", False, 0, v)
            
            F.append([EFN, EFX, EFY, EFZ])
        F = np.array(F, dtype = "float")
        Matrix_Storage(F)

    #Give Option to Pull Matrix for the Reaction Forces

    function_receive = Matrix_Pull("support reaction values")
    
    if function_receive[2] is True:
        
        R = function_receive[0]
        ReF = function_receive[1]
    
    #If the User doesn't want to, let the user create the Support Reaction values
    else:
        
        ReF = Init_Array("reaction forces")
        R = []
        
        for i in range(0, ReF):
            
            v = i + 1
            
            no_overlap = False
            
            while not no_overlap:
                
                RN = Find_Values("reaction force", "node", True, N[:, 0], v)
                RDX = DOF("x-coordinate")
                RDY = DOF("y-coordinate")
                RDZ = DOF("z-coordinate")
                
                if i != 0:
                    no_overlap = Check_Reaction_Duplicates(RN, R, "reaction force")
                
                else:
                    no_overlap = True
            
            R.append([RN, RDX, RDY, RDZ])
        R = np.array(R, dtype = "float")
        Matrix_Storage(R)
    
    #Establishing Classes Depending on 2d or 3d

    if checkif2d(N) is True:
        
        #Establishing the Class of Two-Dimensional Structure
        class two_dimensional_figure:

            def __init__(self, node_array, element_array, forces_array, reaction_array):
                self.node_array, self.element_array, self.forces_array, self.reaction_array = node_array, element_array, forces_array, reaction_array

            #Function for finding the specific positions of the nodes given their IDs
            def pullposnodes(self, IDs):
                
                node_pos_list = []

                for Node_ID in IDs:
                    
                    temp_nodepos_list = []
                    
                    for i in range(1, 3):
                        nodepos = self.node_array[int(Node_ID) - 1, i]
                        temp_nodepos_list.append(nodepos)
                    
                    node_pos_list.append(tuple(temp_nodepos_list))
            
                return node_pos_list

            #Function for finding the angles of the vector
            def model_line(self, n1, n2):
                
                tvec = (n2[0] - n1[0], n2[1] - n1[1])
                angle_ratios = []
                divisor = math.sqrt(tvec[0]**2 + tvec[1]**2)
                
                for i in range(2):
                    angle_ratio = rounding(tvec[i] / divisor)
                    angle_ratios.append(angle_ratio)

                return angle_ratios, divisor
            
            #Function for getting the Conversion Matrix of the Specific Bar
            def getconversionmatrix(self, angle_ratios):

                return np.array([[angle_ratios[0], angle_ratios[1], 0.0, 0.0], [0.0, 0.0, angle_ratios[0], angle_ratios[1]]])

            #Function for Taking a Global Stiffness Matrix for an Element and adding it to the Global Stiffness Matrix
            def assemblingglobalstiffness(self, nodeIDs, globalmatrix, localmatrix):
                
                matrix1, matrix2, matrix3, matrix4 = localmatrix[0:2, 0:2], localmatrix[2:4, 0:2], localmatrix[0:2, 2:4], localmatrix[2:4, 2:4]
                indicies = (int(nodeIDs[0] - 1), int(nodeIDs[1] - 1))
                
                matriceslist = ((indicies[0]*2, indicies[0]*2), (indicies[0]*2, indicies[1]*2), (indicies[1]*2, indicies[0]*2), (indicies[1]*2, indicies[1]*2))
                matriceslistcount = 0
                
                for matrix in (matrix1, matrix2, matrix3, matrix4):
                    globalmatrix[matriceslist[matriceslistcount][0]:matriceslist[matriceslistcount][0]+matrix.shape[0], matriceslist[matriceslistcount][1]:matriceslist[matriceslistcount][1]+matrix.shape[1]] += matrix
                    matriceslistcount += 1

                return globalmatrix
            
            #Assembling the Global Force Matrix While Taking into account Reaction Forces
            def assembleglobalforce(self):
                
                force_list, reaction_list, matrix_position_count = [], [], 0

                for node in self.node_array[:, 0]:

                    templistx, templisty = [], []

                    indicies = np.flatnonzero(node == self.forces_array[:, 0])

                    for index in indicies:
                        
                        templistx.append(self.forces_array[index, 1])
                        templisty.append(self.forces_array[index, 2])
                    
                    reaction_index = np.flatnonzero(node == self.reaction_array[:, 0])
                    temp_list_tuple = (templistx, templisty)

                    if len(reaction_index) == 0:

                        for templist in temp_list_tuple:
                            
                            force_list.append([sum(templist)])
                            matrix_position_count += 1

                    else:
                        
                        for y_index in range(1, 3):
                            
                            if self.reaction_array[reaction_index[0], y_index] == 1.0:
                                
                                reaction_list.append([node, matrix_position_count, sum(temp_list_tuple[y_index - 1])])
                                matrix_position_count += 1

                            else:

                                force_list.append([sum(temp_list_tuple[y_index - 1])])
                                matrix_position_count += 1
                
                return np.array(force_list), np.array(reaction_list)
            
            #Set Up Full Unshaved Global Displacement Matrix
            def setupglobaldisplacement(self, shaveddisplacement, reactionlist):

                matrix = np.ones((len(self.node_array)*2, 1), dtype = "float")

                for node in self.node_array[:, 0]:
                    
                    indicies = np.flatnonzero(node == reactionlist[:, 0])
                    
                    for index in indicies:
                        matrix[int(reactionlist[index, 1]), 0] = 0
                
                mcount, scount = 0, 0
                
                for row in matrix:
                    if row[0] != 0.0:
                        matrix[mcount, 0] = shaveddisplacement[scount, 0]
                        mcount += 1
                        scount += 1
                    else:
                        mcount += 1
                
                return matrix

            #Find the Specific Element's Displacement
            def findelementdisplacement(self, elementid, globaldisplacement):
                matrix = []
                
                nodes = findnodes(elementid, self.element_array)
                
                for i in range(2):
                    matrix.append([globaldisplacement[(int(nodes[0]) - 1)*2 + i][0]])
                
                for i in range(2):
                    matrix.append([globaldisplacement[(int(nodes[1]) - 1)*2 + i][0]])
                
                return np.array(matrix)

            #Create a Numpy Array With the Displaced Nodes
            def displacednodes(self, globaldisplacement):

                displacednodes, addingdisplacementcount = [], 0

                for row in self.node_array:
                    
                    temp = []

                    for value in row[1:3]:
                        
                        temp.append(value + 2*globaldisplacement[addingdisplacementcount, 0])
                        addingdisplacementcount += 1
                    
                    displacednodes.append(temp)
                
                return np.array(displacednodes)
            
            #Seperate Each Displaced Node Array Into Different Matrices For Plotting
            def pltnodematrix(self, displaced):
                
                for element in self.element_array:
                    
                    elementnode1, elementnode2, x, y = element[1], element[2], [], []
                    
                    for elementnode in (elementnode1, elementnode2):
                        
                        count = 0
                        index = np.flatnonzero(elementnode == self.node_array[:, 0])
                        
                        for nodelist in (x, y):
                            
                            nodelist.append(displaced[index, count])
                            count += 1
                
                    plt.plot(x, y, "go--")
            
            #Find The X and Y Limits for Plotting
            def findplotlimit(self):
                
                xlist, ylist = [], []

                for noderow in self.node_array:

                    xlist.append(noderow[1])
                    ylist.append(noderow[2])
                
                max_x_value, max_y_value, min_x_value, min_y_value, max_list, min_list = max(xlist), max(ylist), min(xlist), min(ylist), [], []

                for minvalue in (min_x_value, min_y_value):

                    min_list.append(minvalue - 2)
                
                for maxvalue in (max_x_value, max_y_value):

                    max_list.append(maxvalue + 2)
                
                return max_list, min_list

            #Create the Lists For Plotting the NonDisplaced Points 
            def nondisplacedplotting(self):
                
                for element in self.element_array:
                    
                    elementnode1, elementnode2, nodesplotx, nodesploty = element[1], element[2], [], []
                    
                    for elementnode in (elementnode1, elementnode2):
                        
                        index, nodecount = np.flatnonzero(elementnode == self.node_array[:, 0]), 1
                        
                        for nodelist in (nodesplotx, nodesploty):
                            
                            nodelist.append(self.node_array[index, nodecount])
                            nodecount += 1
                    
                    plt.plot(nodesplotx, nodesploty, "ro--")

            #Plot the External Forces on the Graph
            def plotexternalforces(self):

                xlist, ylist = [], []
                
                for forcenode in self.forces_array[:, 0]:
                    
                    xlist.append(self.node_array[int(forcenode) - 1, 1])
                    ylist.append(self.node_array[int(forcenode) - 1, 2])
                
                xmagnitude, ymagnitude = [], []

                for forcerow in self.forces_array:
                    
                    maxlist = []

                    for i in range(1, 3):
                        
                        maxlist.append(forcerow[i])
                    
                    divisior = max([abs(x) for x in maxlist])
                    
                    xmagnitude.append(forcerow[1] / divisior)
                    ymagnitude.append(forcerow[2] / divisior)

                positioncount = 0

                for xposition in xlist:
                    
                    yposition, xforcemag, yforcemag, totalxforce, totalyforce = ylist[positioncount], xmagnitude[positioncount], ymagnitude[positioncount], self.forces_array[positioncount, 1], self.forces_array[positioncount, 2]
                    positioncount += 1

                    xposition += xforcemag / 2
                    yposition += yforcemag / 2 
                    total_rotation = math.degrees(math.atan2(yforcemag, xforcemag))
                    total_mag = math.sqrt(totalxforce**2 + totalyforce**2)

                    plt.text(xposition, yposition, str(roundingforgraphing(total_mag)) + " N", fontsize = "large", ha = "center", va = "bottom", transform_rotates_text = True, rotation = total_rotation, rotation_mode = "anchor")

                plt.quiver(xlist, ylist, xmagnitude, ymagnitude, angles='xy', scale_units='xy', scale = 1, color = "y")
            
            #Plot the Reaction Forces on the Graph
            def plotreactionforces(self, globalreactions):

                xlist, ylist = [], []

                for reactionnode in self.reaction_array[:, 0]:

                    xlist.append(self.node_array[int(reactionnode) - 1, 1])
                    ylist.append(self.node_array[int(reactionnode) - 1, 2])

                xmagnitude, ymagnitude, textxmag, textymag, globalreactionindex, magnitudecount = [], [], [], [], 0, 0
                
                for reactionrow in self.reaction_array:

                    reactionrowcount = 1
                    for reactionlist in (xmagnitude, ymagnitude):
                        
                        reactiondirection = reactionrow[reactionrowcount]
                        reactionrowcount += 1

                        if reactiondirection == 1.0:
                            
                            reactionlist.append(globalreactions[globalreactionindex][1])
                            globalreactionindex += 1
                        
                        else:
                            
                            reactionlist.append(0)

                    textxmag.append(xmagnitude[magnitudecount])
                    textymag.append(ymagnitude[magnitudecount])
                    
                    divisor = max([abs(x) for x in [xmagnitude[magnitudecount], ymagnitude[magnitudecount]]])
                    if divisor != 0:

                        for magnitude in (xmagnitude, ymagnitude):

                            magnitude[magnitudecount] = magnitude[magnitudecount] / divisor
                    
                    magnitudecount += 1

                positioncount = 0
                
                for xposition in xlist:

                    yposition, xforcemag, yforcemag, totalxforce, totalyforce = ylist[positioncount], xmagnitude[positioncount], ymagnitude[positioncount], textxmag[positioncount], textymag[positioncount]
                    positioncount += 1
                    total_mag = math.sqrt(totalxforce**2 + totalyforce**2)

                    if total_mag != 0.0:

                        xposition += xforcemag / 2
                        yposition += yforcemag / 2
                        total_rotation = math.degrees(math.atan2(yforcemag, xforcemag))
                        
                        plt.text(xposition, yposition, str(roundingforgraphing(total_mag)) + " N", fontsize = "large", ha = "center", va = "bottom", transform_rotates_text = True, rotation = total_rotation, rotation_mode = "anchor")

                plt.quiver(xlist, ylist, xmagnitude, ymagnitude, angles='xy', scale_units='xy', scale = 1, color = "c")

            #Plot the Tensions/Compressions on the Graph with Text
            def plottensioncompression(self, globalelementforces):
                
                for elementid in self.element_array[:, 0]:

                    node1pos, node2pos = int(self.element_array[int(elementid) - 1, 1]) - 1, int(self.element_array[int(elementid) - 1, 2]) - 1
                    nodeposx, nodeposy = (self.node_array[node1pos, 1], self.node_array[node2pos, 1]), (self.node_array[node1pos, 2], self.node_array[node2pos, 2])
                    finalposx, finalposy = nodeposx[0] + (nodeposx[1] - nodeposx[0]) / 2, nodeposy[0] + (nodeposy[1] - nodeposy[0]) / 2

                    if nodeposx[0] > nodeposx[1]:
                        
                        totalrotation = math.degrees(math.atan2(nodeposy[0] - nodeposy[1], nodeposx[0] - nodeposx[1]))
                    
                    else:

                        totalrotation = math.degrees(math.atan2(nodeposy[1] - nodeposy[0], nodeposx[1] - nodeposx[0]))

                    leftforce, rightforce = roundingforgraphing(globalelementforces[int(elementid) - 1, 0]), roundingforgraphing(globalelementforces[int(elementid) - 1, 1])
                    
                    if leftforce < 0:
                        
                        plt.text(finalposx, finalposy, str(rightforce) + " (t)", fontsize = "large", ha = "center", va = "bottom", transform_rotates_text = True, rotation = totalrotation, rotation_mode = "anchor")
                    
                    elif leftforce > 0:

                        plt.text(finalposx, finalposy, str(leftforce) + " (c)", fontsize = "large", ha = "center", va = "bottom", transform_rotates_text = True, rotation = totalrotation, rotation_mode = "anchor")
                    
                    else:

                        plt.text(finalposx, finalposy, "0 force member", fontsize = "large", ha = "center", va = "bottom", transform_rotates_text = True, rotation = totalrotation, rotation_mode = "anchor")

        user_fig = two_dimensional_figure(N, E, F, R)
        globalstiffnessmatrix = np.zeros((Nt*2, Nt*2), dtype = "float")
    
    else:

        #Establishing the Class of Three-Dimensional Structure
        class three_dimensional_figure:
            
            def __init__(self, node_array, element_array, forces_array, reaction_array):
                self.node_array, self.element_array, self.forces_array, self.reaction_array = node_array, element_array, forces_array, reaction_array
            
            def pullposnodes(self, IDs):
                
                node_pos_list = []

                for Node_ID in IDs:
                    
                    temp_nodepos_list = []
                    
                    for i in range(1, 4):
                        nodepos = self.node_array[int(Node_ID) - 1, i]
                        temp_nodepos_list.append(nodepos)
                    
                    node_pos_list.append(tuple(temp_nodepos_list))
            
                return node_pos_list
            
            def model_line(self, n1, n2):
                
                tvec = (n2[0] - n1[0], n2[1] - n1[1], n2[2] - n1[2])
                angle_ratios = []
                divisor = math.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)
                
                for i in range(3):
                    angle_ratio = rounding(tvec[i] / divisor)
                    angle_ratios.append(angle_ratio)
                
                return angle_ratios, divisor
            
            def getconversionmatrix(self, angle_ratios):

                return np.array([[angle_ratios[0], angle_ratios[1], angle_ratios[2], 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, angle_ratios[0], angle_ratios[1], angle_ratios[2]]])
            
            def assemblingglobalstiffness(self, nodeIDs, globalmatrix, localmatrix):
                
                matrix1, matrix2, matrix3, matrix4 = localmatrix[0:3, 0:3], localmatrix[3:6, 0:3], localmatrix[0:3, 3:6], localmatrix[3:6, 3:6]
                indicies = (int(nodeIDs[0]) - 1, int(nodeIDs[1]) - 1)

                matriceslist = ((indicies[0]*3, indicies[0]*3), (indicies[0]*3, indicies[1]*3), (indicies[1]*3, indicies[0]*3), (indicies[1]*3, indicies[1]*3))
                matriceslistcount = 0
                
                for matrix in (matrix1, matrix2, matrix3, matrix4):
                    globalmatrix[matriceslist[matriceslistcount][0]:matriceslist[matriceslistcount][0]+matrix.shape[0], matriceslist[matriceslistcount][1]:matriceslist[matriceslistcount][1]+matrix.shape[1]] += matrix
                    matriceslistcount = matriceslistcount + 1
                
                return globalmatrix

            def assembleglobalforce(self):
                
                force_list, reaction_list, matrix_position_count = [], [], 0

                for node in self.node_array[:, 0]:

                    templistx, templisty, templistz = [], [], []

                    indicies = np.flatnonzero(node == self.forces_array[:, 0])

                    for index in indicies:
                        
                        templistx.append(self.forces_array[index, 1])
                        templisty.append(self.forces_array[index, 2])
                        templistz.append(self.forces_array[index, 3])

                    reaction_index = np.flatnonzero(node == self.reaction_array[:, 0])
                    temp_list_tuple = (templistx, templisty, templistz)

                    if len(reaction_index) == 0:

                        for templist in temp_list_tuple:
                            
                            force_list.append([sum(templist)])
                            matrix_position_count += 1

                    else:
                        
                        for y_index in range(1, 4):
                            
                            if self.reaction_array[reaction_index[0], y_index] == 1.0:
                                
                                reaction_list.append([node, matrix_position_count, sum(temp_list_tuple[y_index - 1])])
                                matrix_position_count += 1

                            else:

                                force_list.append([sum(temp_list_tuple[y_index - 1])])
                                matrix_position_count += 1
                
                return np.array(force_list), np.array(reaction_list)
    
            def setupglobaldisplacement(self, shaveddisplacement, reactionlist):

                matrix = np.ones((len(self.node_array)*3, 1), dtype = "float")

                for node in self.node_array[:, 0]:
                    
                    indicies = np.flatnonzero(node == reactionlist[:, 0])
                    
                    for index in indicies:
                        matrix[int(reactionlist[index, 1]), 0] = 0
                
                mcount, scount = 0, 0
                
                for row in matrix:
                    if row[0] != 0.0:
                        matrix[mcount, 0] = shaveddisplacement[scount, 0]
                        mcount += 1
                        scount += 1
                    else:
                        mcount += 1
                
                return matrix
    
            def findelementdisplacement(self, elementid, globaldisplacement):
                matrix = []
                
                nodes = findnodes(elementid, self.element_array)
                
                for i in range(3):
                    matrix.append([rounding(globaldisplacement[(int(nodes[0]) - 1)*3 + i][0])])
                
                for i in range(3):
                    matrix.append([rounding(globaldisplacement[(int(nodes[1]) - 1)*3 + i][0])])
                
                return np.array(matrix)
    
            def displacednodes(self, globaldisplacement):

                displacednodes, addingdisplacementcount = [], 0

                for row in self.node_array:
                    
                    temp = []

                    for value in row[1:4]:
                        
                        temp.append(value + 2*globaldisplacement[addingdisplacementcount, 0])
                        addingdisplacementcount += 1
                    
                    displacednodes.append(temp)
                
                return np.array(displacednodes)
            
            def pltnodematrix(self, displaced):

                for element in self.element_array:
                    
                    elementnode1, elementnode2, x, y, z = element[1], element[2], [], [], []
                    
                    for elementnode in (elementnode1, elementnode2):
                        
                        count = 0
                        index = np.flatnonzero(elementnode == self.node_array[:, 0])
                        
                        for nodelist in (x, y, z):
                            
                            nodelist.append(displaced[index, count])
                            count += 1
                    
                    plt.plot(x, y, z, "go--")
            
            def nondisplacedplotting(self):
                
                for element in self.element_array:
                    
                    elementnode1, elementnode2, nodesplotx, nodesploty, nodesplotz = element[1], element[2], [], [], []
                    
                    for elementnode in (elementnode1, elementnode2):
                        
                        index = np.flatnonzero(elementnode == self.node_array[:, 0])
                        nodecount = 1
                        
                        for nodelist in (nodesplotx, nodesploty, nodesplotz):
                            
                            nodelist.append(self.node_array[index, nodecount])
                            nodecount += 1
                    
                    plt.plot(nodesplotx, nodesploty, nodesplotz, "ro--")

            def findplotlimit(self):
                
                xlist, ylist, zlist = [], [], []

                for noderow in self.node_array:

                    xlist.append(noderow[1])
                    ylist.append(noderow[2])
                    zlist.append(noderow[3])
                
                max_x_value, max_y_value, min_x_value, min_y_value, max_z_value, min_z_value, max_list, min_list = max(xlist), max(ylist), min(xlist), min(ylist), max(zlist), min(zlist), [], []

                for minvalue in (min_x_value, min_y_value, min_z_value):

                    min_list.append(minvalue - 2)
                
                for maxvalue in (max_x_value, max_y_value, max_z_value):

                    max_list.append(maxvalue + 2)
                
                return max_list, min_list

            def plotexternalforces(self, ax):

                xlist, ylist, zlist, xmagnitude, ymagnitude, zmagnitude = [], [], [], [], [], []
                
                for forcenode in self.forces_array[:, 0]:
                    
                    xlist.append(self.node_array[int(forcenode) - 1, 1])
                    ylist.append(self.node_array[int(forcenode) - 1, 2])
                    zlist.append(self.node_array[int(forcenode) - 1, 3])

                for forcerow in self.forces_array:
                    
                    maxlist = []

                    for i in range(1, 4):
                        
                        maxlist.append(forcerow[i])
                    
                    divisior = max([abs(x) for x in maxlist])
                    
                    xmagnitude.append(forcerow[1] / divisior)
                    ymagnitude.append(forcerow[2] / divisior)
                    zmagnitude.append(forcerow[3] / divisior)
                
                positioncount = 0

                for xposition in xlist:
                    yposition, zposition, xmag, ymag, zmag, plotmag = ylist[positioncount], zlist[positioncount], xmagnitude[positioncount], ymagnitude[positioncount], zmagnitude[positioncount], math.sqrt(self.forces_array[positioncount, 1]**2 + self.forces_array[positioncount, 2]**2 + self.forces_array[positioncount, 3]**2)
                    positioncount += 1

                    for position, mag in zip((xposition, yposition, zposition), (xmag, ymag, zmag)):

                        position += mag / 2
                    zdir = (xmag, ymag, zmag)

                    ax.text(xposition, yposition, zposition, str(roundingforgraphing(plotmag)) + " N", zdir, fontsize = "small")
                       
                plt.quiver(xlist, ylist, zlist, xmagnitude, ymagnitude, zmagnitude, arrow_length_ratio = 1, color = "y")
    
            def plotreactionforces(self, globalreactions, ax):

                xlist, ylist, zlist = [], [], []

                for reactionnode in self.reaction_array[:, 0]:

                    xlist.append(self.node_array[int(reactionnode) - 1, 1])
                    ylist.append(self.node_array[int(reactionnode) - 1, 2])
                    zlist.append(self.node_array[int(reactionnode) - 1, 3])

                xmagnitude, ymagnitude, zmagnitude, textxmag, textymag, textzmag, globalreactionindex, magnitudecount = [], [], [], [], [], [], 0, 0
                
                for reactionrow in self.reaction_array:

                    reactionrowcount = 1
                    for reactionlist in (xmagnitude, ymagnitude, zmagnitude):
                        
                        reactiondirection = reactionrow[reactionrowcount]
                        reactionrowcount += 1

                        if reactiondirection == 1.0:
                            
                            reactionlist.append(globalreactions[globalreactionindex][1])
                            globalreactionindex += 1
                        
                        else:
                            
                            reactionlist.append(0)

                    for textmag, mag in zip((textxmag, textymag, textzmag), (xmagnitude, ymagnitude, zmagnitude)):

                        textmag.append(mag[magnitudecount])

                    divisor = max([abs(x) for x in [xmagnitude[magnitudecount], ymagnitude[magnitudecount], zmagnitude[magnitudecount]]])
                    if divisor != 0:
                        
                        for magnitude in (xmagnitude, ymagnitude, zmagnitude):

                            magnitude[magnitudecount] = magnitude[magnitudecount] / divisor
                    
                    magnitudecount += 1
                
                positioncount = 0

                for xreaction, yreaction, zreaction, xposition, yposition, zposition in zip(xmagnitude, ymagnitude, zmagnitude, xlist, ylist, zlist):

                    for position, reaction in zip((xposition, yposition, zposition), (xreaction, yreaction, zreaction)):

                        position += reaction / 2
                    
                    total_mag = math.sqrt(textxmag[positioncount]**2 + textymag[positioncount]**2 + textzmag[positioncount]**2)
                    positioncount += 1

                    if total_mag != 0.0:
                        
                        ax.text(xposition, yposition, zposition, str(roundingforgraphing(total_mag)) + " N", (xreaction, yreaction, zreaction), fontsize = "small")

                plt.quiver(xlist, ylist, zlist, xmagnitude, ymagnitude, zmagnitude, arrow_length_ratio = 1, color = "c")

            def plottensioncompression(self, globalelementforces, ax):

                for elementid in self.element_array[:, 0]:

                    node1pos, node2pos = int(self.element_array[int(elementid) - 1, 1]) - 1, int(self.element_array[int(elementid) - 1, 2]) - 1
                    nodeposx, nodeposy, nodeposz = (self.node_array[node1pos, 1], self.node_array[node2pos, 1]), (self.node_array[node1pos, 2], self.node_array[node2pos, 2]), (self.node_array[node1pos, 3], self.node_array[node2pos, 3])
                    finalposx, finalposy, finalposz = nodeposx[0] + (nodeposx[1] - nodeposx[0]) / 2, nodeposy[0] + (nodeposy[1] - nodeposy[0]) / 2, nodeposz[0] + (nodeposz[1] - nodeposz[0]) / 2
                    
                    if nodeposx[0] > nodeposx[1]:

                        rotatex, rotatey, rotatez = nodeposx[0] - nodeposx[1], nodeposy[0] - nodeposy[1], nodeposz[0] - nodeposz[1]
                    
                    else:

                        rotatex, rotatey, rotatez = nodeposx[1] - nodeposx[0], nodeposy[1] - nodeposy[0], nodeposz[1] - nodeposz[0]
                    
                    leftforce, rightforce = roundingforgraphing(globalelementforces[int(elementid) - 1, 0]), roundingforgraphing(globalelementforces[int(elementid) - 1, 1])

                    if leftforce < 0:

                        ax.text(finalposx, finalposy, finalposz, str(rightforce) + " (t)", (rotatex, rotatey, rotatez), fontsize = "small", ha = "center", va = "bottom")
                    
                    elif leftforce > 0:

                        ax.text(finalposx, finalposy, finalposz, str(leftforce) + " (c)", (rotatex, rotatey, rotatez), fontsize = "small", ha = "center", va = "bottom")
                    
                    else:

                        ax.text(finalposx, finalposy, finalposz, "0 force member", (rotatex, rotatey, rotatez), fontsize = "small", ha = "center", va = "bottom")

        user_fig = three_dimensional_figure(N, E, F, R)
        globalstiffnessmatrix = np.zeros((Nt*3, Nt*3), dtype = "float")
    
    #Applying the Stiffness Method

    localconversions, localstiffnesses = [], []
    Create_Element_Matrix(E)
    ranelementmatrix, elementlist = Utilize_Element_Matrix(E)

    for elementid in E[:, 0]:
        
        nodetuple = findnodes(elementid, E)
        nodepos = user_fig.pullposnodes(nodetuple)
        Conversion_Matrix = user_fig.getconversionmatrix(user_fig.model_line(nodepos[0], nodepos[1])[0])

        if ranelementmatrix is True:

            localstiffness = stiffness(elementlist[int(elementid) - 1], user_fig.model_line(nodepos[0], nodepos[1])[1])
        
        else:

            localstiffness = stiffness(findingcomponents(str(elementid), "Young's Modulus multiplied by Cross-sectional Area"), user_fig.model_line(nodepos[0], nodepos[1])[1])
        
        localstiffnesses.append(localstiffness)
        localstiffness = converttoglobalstiffness(Conversion_Matrix, localstiffness)
        globalstiffnessmatrix = user_fig.assemblingglobalstiffness(nodetuple, globalstiffnessmatrix, localstiffness)
        localconversions.append(Conversion_Matrix)

    shavedglobalforcematrix, rlist = user_fig.assembleglobalforce()
    
    shavedstiffness = removepartsglobalstiffness(globalstiffnessmatrix, rlist)
    shaveddisplacements = finddisplacements(shavedstiffness, shavedglobalforcematrix)
    
    globaldisplacement = user_fig.setupglobaldisplacement(shaveddisplacements, rlist)
    globalreactions = findreactionforces(globalstiffnessmatrix, rlist, globaldisplacement)
    
    globalelementforces = []
    
    for elementid in E[:, 0]:
        
        elementdisplacement = user_fig.findelementdisplacement(elementid, globaldisplacement)
        globalelementforces.append(findelementforce(elementdisplacement, localstiffnesses[int(elementid) - 1], localconversions[int(elementid) - 1]))
    
    globalelementforces = np.atleast_2d(np.array(globalelementforces))
    print("This is the global displacement matrix: ")
    print(globaldisplacement)
    print("This is the global reaction matrix: ")
    print(globalreactions)
    print("This is the global element force matrix: ")
    print(globalelementforces)

    #Plotting

    if checkif2d(N) is True:
        
        ax = plt.axes()
        function_receive = user_fig.findplotlimit()
        plt.xlim(function_receive[1][0], function_receive[0][0])
        plt.ylim(function_receive[1][1], function_receive[0][1])

        user_fig.pltnodematrix(user_fig.displacednodes(globaldisplacement))
        user_fig.nondisplacedplotting()

        user_fig.plotexternalforces()
        user_fig.plotreactionforces(globalreactions)
        user_fig.plottensioncompression(globalelementforces)
    
    else:

        ax = plt.axes(projection = "3d")
        function_receive = user_fig.findplotlimit()
        plt.xlim(function_receive[1][0], function_receive[0][0])
        plt.ylim(function_receive[1][1], function_receive[0][1])
        ax.set_zlim(function_receive[1][2], function_receive[0][2])
        
        user_fig.pltnodematrix(user_fig.displacednodes(globaldisplacement))
        user_fig.nondisplacedplotting()

        user_fig.plotexternalforces(ax)
        user_fig.plotreactionforces(globalreactions, ax)
        user_fig.plottensioncompression(globalelementforces, ax)
    
    plt.show()

main()