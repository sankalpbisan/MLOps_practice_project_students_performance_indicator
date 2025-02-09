from setuptools import find_packages, setup  #This finds all the packages in entire the dir (ML project dir)

hyphen_dot_e = '-e .' #This line (used in requirements.txt) helps in running the setup.py file
        #This will automatically trigger setup.py file
def get_requirements(file_path:str) -> list[str]:
    '''
    This function will return the list of requirements (libraries etc.)
    '''
    requirements = []
    with open(file_path,'r') as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        print(requirements)

    if hyphen_dot_e in requirements:
        requirements.remove(hyphen_dot_e)
        # -e . isn't any packages we are using it for our convenience
        # So, it should not be considered as package
    print(requirements)
    return requirements


setup(
    name = 'mlops_roject',
    version = '0.0.1',
    author = 'Sankalp',
    author_email = 'sankalpbisan07@gmail.com',
    packages = find_packages(),
#    install_requires = ['pandas','numpy','seaborn','sklearn']
    #It isn't feasible to write all libraries in here so we'll create a function for it
    install_requires = get_requirements('requirements.txt')

#It will find and consider files as packages which has a __init.py__ file in some other dir, lets say "src"
    #Once it finds it, it will build that particular package and can be used as library just like other
)
