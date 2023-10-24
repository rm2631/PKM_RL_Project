from pyboy import PyBoy
from pyboy import WindowEvent
from datetime import datetime

class PKM_PyBoy():
    command_map = {
        "UP": [WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP],
        "DOWN": [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN],
        "LEFT": [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
        "RIGHT": [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
        "A": [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A],
        "B": [WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B],
    }

    def __init__(self):
        self.pyboy = PyBoy('ROMs/Pokemon Red.gb')
        state =  open("ROMs/Pokemon Red.gb.state", "rb")
        self.pyboy.load_state(state)
        
        while not self.pyboy.tick():
            pass
        self.pyboy.stop()

    def send_command(self, command):
        if command in self.command_map:
            self.pyboy.send_input(self.command_map[command][0])
            self.pyboy.tick()
            self.pyboy.send_input(self.command_map[command][1])
            self._get_screen()

    def _get_screen(self):
        pil_image = self.pyboy.screen_image()
        #save image with datetime as name
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        pil_image.save("screenshots/" + dt_string + ".png")
            
if __name__ == '__main__':
    pyboy = PKM_PyBoy()