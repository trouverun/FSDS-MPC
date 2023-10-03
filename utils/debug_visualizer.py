import numpy as np
import config
import pyqtgraph as pg
import traceback
import time
from queue import Empty
from PyQt6 import QtCore
from utils.drawing import draw_detections, draw_global_map


class Visualizer(pg.GraphicsView):
    def __init__(self, perception_queue, controller_data_queue, exit_event):
        super(Visualizer, self).__init__()
        res = int(config.camera_resolution / 3)
        self.perception_queue = perception_queue
        self.controller_data_queue = controller_data_queue
        self.exit_event = exit_event

        self.backwards_horizon = None
        self.initialized = False

        self.layout = pg.GraphicsLayout()
        self.setCentralWidget(self.layout)
        self.vb = self.layout.addViewBox(lockAspect=True, invertY=True)

        self.images = [np.ones([res, res, 3])]*2
        self.image_item = pg.ImageItem()
        self.vb.addItem(self.image_item)
        self.layout.nextColumn()

        plot_list = [
            ("speed", 0, config.car_max_speed, "m/s", [
                ((255, 255, 0), "vx")
            ]),
            ("w", -180, 180, "deg/s", [
                ((255, 255, 0), "hdg"),
                ((0, 255, 255), "w")
            ]),
            ("controls", -1, 1, "amplitude", [
                ((255, 255, 0), "steer"),
                ((0, 255, 255), "throttle"),
            ]),
            ("vision duration", 0, 100, "ms", [
                ((255, 255, 0), "duration"),
                ((0, 255, 255), "target")
            ]),
            ("mpc duration", 0, 100, "ms", [
                ((255, 255, 0), "duration"),
                ((0, 255, 255), "target")
            ])
        ]

        # Construct graph:
        self.n_plot_items = sum([len(t[4]) for t in plot_list])
        self.plot_history_len = 100
        self.x = np.repeat(np.arange(self.plot_history_len), self.n_plot_items).reshape(self.plot_history_len, self.n_plot_items).T
        self.y = np.zeros([self.n_plot_items, self.plot_history_len])
        self.plots = []
        self.plot_items = []
        plot_i = 0
        item_i = 0
        for title, lb, ub, y_label, items in plot_list:
            self.plots.append(self.layout.addPlot(row=plot_i+1, col=0))
            self.layout.nextRow()
            self.plots[plot_i].setYRange(lb, ub)
            self.plots[plot_i].setTitle(title)
            self.plots[plot_i].showGrid(True, True)
            self.plots[plot_i].setLabel(axis='left', text=y_label)
            self.plots[plot_i].setLabel(axis='bottom', text='Time since (s)')
            # if plot_i == 4:
            #     self.plots[plot_i].setFixedHeight(400)
            legend = self.plots[plot_i].addLegend()
            legend.setBrush('k')
            legend.setOffset(1)
            for color, name in items:
                self.plot_items.append(
                    self.plots[plot_i].plot(
                        self.x[item_i], self.y[item_i], pen=pg.mkPen(color=color), name=name
                    )
                )
                item_i += 1
            plot_i += 1

        self.n_plots = plot_i

        self.layout.layout.setRowFixedHeight(0, res)
        self.layout.layout.setColumnFixedWidth(0, 2*res)
        for i in range(plot_i):
            self.layout.layout.setRowFixedHeight(i+1, 180)
            self.layout.layout.setColumnFixedWidth(i+1, 2*res)

        height = res + int((plot_i+1)*(180+self.layout.layout.rowSpacing(0))) + 20
        self.setFixedHeight(height)
        width = 2*res + int(2*self.layout.layout.columnSpacing(0)) + 20
        self.setFixedWidth(width)

        # Start UI tick:
        self.timer = QtCore.QTimer()
        self.timer.setInterval(5)
        self.timer.timeout.connect(self.update_data)
        self.timer.start()

    def update_data(self):
        if self.exit_event.is_set():
            self.close()

        try:
            perception_data = self.perception_queue.get(block=False)

            # Extract graph data:
            sum = perception_data.camera_ms + perception_data.yolo_ms + perception_data.kp_ms + perception_data.pnp_ms + perception_data.map_update_ms
            graph_data = {
                'perception timing': (5, [sum, 1e3*1/30]),
            }

            # Update and draw the odometry/info graph
            for first_i, values_list in graph_data.values():
                for j, value in enumerate(values_list):
                    self.y[first_i+j, 0:-1] = self.y[first_i+j, 1:]
                    self.y[first_i+j, -1] = value
                    self.plot_items[first_i+j].setData(self.x[first_i+j], self.y[first_i+j])

            frame = draw_detections(perception_data.camera_frame, perception_data.blue_pixels, perception_data.yellow_pixels)
            self.images[1] = frame
        except Empty:
            pass
        except Exception as e:
            print("Graph error: ", traceback.print_exception(e))
            self.close()

        try:
            controller_data = self.controller_data_queue.get(block=False)

            graph_data = {
                'speed': (0, [controller_data.car_state[3]]),
                'w': (1, [180/np.pi * controller_data.car_state[2], 180/np.pi * controller_data.car_state[5]]),
                'controls': (3, [controller_data.car_state[-3], controller_data.car_state[-2]]),
                'mpc timing': (7, [controller_data.solver_time_ms, 1e3*config.mpc_dt])
            }

            # Update and draw the odometry/info graph
            for first_i, values_list in graph_data.values():
                for j, value in enumerate(values_list):
                    self.y[first_i+j, 0:-1] = self.y[first_i+j, 1:]
                    self.y[first_i+j, -1] = value
                    self.plot_items[first_i+j].setData(self.x[first_i+j], self.y[first_i+j])

            frame = draw_global_map(
                controller_data.car_state, controller_data.blue_cones, controller_data.yellow_cones, controller_data.midpoints,
                controller_data.trajectory, np.zeros([config.mpc_horizon, 3])
            )
            self.images[0] = frame

        except Empty:
            pass
        except Exception as e:
            print("Graph error: ", traceback.print_exception(e))
            self.close()

        self.image_item.setImage(np.vstack(self.images))

