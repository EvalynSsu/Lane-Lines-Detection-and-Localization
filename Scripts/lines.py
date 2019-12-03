import numpy as np

class lines:

    maxNum = 50
    threshold = 1
    insist = True

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None




    def add_rst(self, detected, fit, radius, bias, linepix, frame):

        resonableCurve = self.isReasonable(fit)

        if resonableCurve == False:
            self.insist = False

        else:

            # for starting 50 is to init
            self.recent_xfitted.append(linepix)
            multiplier = min(frame, self.maxNum)

            if frame < 2:
                self.bestx =linepix
                self.best_fit = fit
                self.radius_of_curvature = radius

            else:

                self.insist = True

                for index in range(0,2):
                    diff = self.best_fit[0][index] - fit[0][index]
                    if abs(diff)>self.threshold:
                        self.insist = False
                        print("\n [Huge Jump] left not inconsist! Redetecting!", index)

                for index in range(0,2):
                    diff = self.best_fit[1][index] - fit[1][index]
                    if abs(diff)>self.threshold:
                        self.insist = False
                        print("\n [Huge Jump] right not insist! Redetecting!", index)

                self.bestx = (self.bestx*multiplier+linepix)/(multiplier+1)
                self.best_fit = ((self.best_fit[0]*multiplier+fit[0])/(multiplier+1), (self.best_fit[1]*multiplier+fit[1])/(multiplier+1))
                self.radius_of_curvature = (self.radius_of_curvature*multiplier+radius)/(multiplier+1)

            if frame > self.maxNum:
                self.recent_xfitted.pop(0)

            self.line_base_pos = bias
            self.current_fit = fit

        return self.insist  # return False to redetect

    def isReasonable(self, fit):

        # check left and right parrell
        diff = abs(fit[0][0]-fit[1][0])
        if diff > 0.01:
            print("\n [OUTLIERS] NOT PARRELL! Discarding")
            return False

        # check if curl too much
        if max(abs(fit[0][0]), abs(fit[1][0])) > 0.01:
            print("\n [OUTLIERS] CRUL TOO MUCH! Discarding")
            return False

        return True

    def smooth(self):
        pass



