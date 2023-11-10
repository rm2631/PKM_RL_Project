from pyboy import PyBoy

pyboy = PyBoy("ROMs/Pokemon Red.gb")
pyboy.set_emulation_speed(3)
while not pyboy.tick():
    pass
pyboy.stop()
