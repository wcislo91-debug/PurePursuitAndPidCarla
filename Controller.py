def compute_yv(self, x, y, yaw, x_l, y_l):
        dx = x_l - x
        dy = y_l - y

        # Transform into vehicle frame
        x_v =  dx * math.cos(yaw) + dy * math.sin(yaw)
        y_v = -dx * math.sin(yaw) + dy * math.cos(yaw)

        return y_v


    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x         = self._current_x
        y         = self._current_y
        yaw       = self._current_yaw
        v         = self._current_speed
        t         = self._current_timestamp
        waypoints = self._waypoints

        throttle_output = 0.0
        brake_output    = 0.0
        steer_output    = 0.0

        # Update desired speed from waypoints
        self.update_desired_speed()
        v_desired = self._desired_speed

        # Skip the first frame to store previous values properly
        if self._start_control_loop and len(waypoints) >= 2:

            ######################################################
            # LONGITUDINAL CONTROLLER — PID ON SPEED
            ######################################################
            Kp = 0.5
            Ki = 0.1
            Kd = 0.05

            # 1) Compute dt
            dt = t - self.vars.t_prev
            if dt <= 0.0:
                dt = 1e-3

            # 2) Speed error
            v_error = v_desired - v

            # 3) Integral term with anti-windup (simple clamping)
            self.vars.v_error_int += v_error * dt
            self.vars.v_error_int = max(min(self.vars.v_error_int, 10.0), -10.0)

            # 4) PID output = desired acceleration
            v_error_der = (v_error - self.vars.v_error_prev) / dt
            a_cmd = Kp * v_error + Ki * self.vars.v_error_int + Kd * v_error_der

            # 5) Convert acceleration to throttle/brake
            if a_cmd >= 0.0:
                throttle_output = min(a_cmd, 1.0)
                brake_output    = 0.0
            else:
                throttle_output = 0.0
                brake_output    = min(-a_cmd, 1.0)

            # 6) Store previous values for next iteration
            self.vars.v_error_prev = v_error
            self.vars.t_prev       = t

            ######################################################
            # LATERAL CONTROLLER — PURE PURSUIT
            ######################################################
            wheelbase = self._wheelbase
            k_ld      = 1.0    # lookahead gain
            Ld_min    = 3.0    # minimum lookahead distance

            total_wp = len(waypoints)

            # 1) FIND CLOSEST WAYPOINT (search forward from last index)
            closest_idx  = self.vars.closest_idx_prev
            closest_dist = float("inf")

            # ensure starting index is valid
            if closest_idx < 0 or closest_idx >= total_wp:
                closest_idx = 0

            for i in range(closest_idx, total_wp):
                dx = waypoints[i][0] - x
                dy = waypoints[i][1] - y
                dist = math.hypot(dx, dy)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx  = i

            # clamp closest index so we always have a point ahead
            if closest_idx >= total_wp - 2:
                closest_idx = max(total_wp - 2, 0)

            self.vars.closest_idx_prev = closest_idx

            # 2) CHOOSE LOOKAHEAD WAYPOINT
            Ld = max(Ld_min, k_ld * max(v, 0.0))  # speed‑dependent lookahead

            lookahead_idx = closest_idx
            while lookahead_idx < total_wp - 1:
                dx = waypoints[lookahead_idx][0] - x
                dy = waypoints[lookahead_idx][1] - y
                if math.hypot(dx, dy) >= Ld:
                    break
                lookahead_idx += 1

            if lookahead_idx >= total_wp:
                lookahead_idx = total_wp - 1

            x_l = waypoints[lookahead_idx][0]
            y_l = waypoints[lookahead_idx][1]

            # 3) TRANSFORM LOOKAHEAD POINT TO VEHICLE FRAME
            y_v = self.compute_yv(x, y, yaw, x_l, y_l)

            # 4) PURE PURSUIT STEERING
            Ld = math.hypot(x_l - x, y_l - y)
            if Ld < 1e-3:
                Ld = 1e-3  # avoid division by zero

            delta = math.atan2(2.0 * wheelbase * y_v, Ld**2)

            # 5) SATURATE STEERING
            max_steer = 1.1  # rad
            delta = max(min(delta, max_steer), -max_steer)

            steer_output = delta
