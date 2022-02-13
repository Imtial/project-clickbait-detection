import { ThemeProvider } from "@emotion/react";
import {
  Button,
  Card,
  CardContent,
  CardActions,
  Container,
  Grid,
  TextField,
  Typography,
  FormControl,
  FormControlLabel,
  FormLabel,
  RadioGroup,
  Radio,
} from "@mui/material";
import { Box } from "@mui/system";
import { createTheme } from "@mui/material/styles";
import { useState } from "react";
import { getPredictions, sendFeedback } from "./services/classifierService";

const theme = createTheme({
  palette: {
    primary: {
      main: "#f5f5f5",
      light: "#ffffff",
      dark: "#c2c2c2",
    },
  },
});

function App() {
  const [inputText, setInputText] = useState("");
  const [headline, setHeadline] = useState("");
  const [className, setClassName] = useState("");
  const [guess, setGuess] = useState("");

  const clickbaitStyle = {
    backgroundColor: "#ff8a80",
    width: "220px",
    height: "200px",
    padding: "20px",
    marginTop: "30px",
  };

  const newsStyle = {
    backgroundColor: "#00e676",
    width: "220px",
    height: "200px",
    padding: "20px",
    marginTop: "30px",
  };

  const handleSubmit = () => {
    const data = sendFeedback({
      headline: headline,
      className: guess,
    });
    if (data) {
      setGuess("");
    }
  };

  return (
    <Container maxWidth="md">
      <Grid container m={5} spacing={2}>
        <Grid item xs={12}>
          <Box display="flex" justifyContent="center">
            <Box margin="auto">
              <Typography align="center" gutterBottom variant="h6">
                CSE472 Project
              </Typography>
              <Typography align="center" variant="h2">
                Clickbait Detection
              </Typography>
              <Typography align="center" gutterBottom variant="subtitle1">
                1605008, 1605017
              </Typography>
            </Box>
          </Box>
        </Grid>
        <Grid item xs={12}>
          <TextField
            variant="outlined"
            label="Headline or URL"
            fullWidth
            multiline
            maxRows={8}
            onChange={(e) => {
              setInputText(e.target.value);
            }}
          />
        </Grid>
        <Grid item xs={12} align="center">
          <Button
            variant="contained"
            size="large"
            onClick={async () => {
              setHeadline("");
              setClassName("");
              const predictions = await getPredictions(inputText);
              console.log(predictions);
              setHeadline(predictions.headline);
              setClassName(predictions.className);
            }}
          >
            Test
          </Button>
        </Grid>
        {className && className.length > 0 && (
          <>
            <Grid item xs={12} mt={3}>
              <Typography variant="h4">{headline}</Typography>
            </Grid>
            <Grid item xs={12}>
              <Box display="flex">
                <Box m="auto">
                  <ThemeProvider theme={theme}>
                    <Card
                      sx={
                        className.toLowerCase() === "news"
                          ? newsStyle
                          : clickbaitStyle
                      }
                    >
                      <CardContent>
                        <Typography variant="h5" gutterBottom>
                          {className}
                        </Typography>
                      </CardContent>
                      <CardActions>
                        <FormControl>
                          <FormLabel>What's your guess?</FormLabel>
                          <RadioGroup
                            row
                            m={2}
                            onChange={(e) => {
                              setGuess(e.target.value);
                            }}
                          >
                            <FormControlLabel
                              value="news"
                              control={<Radio />}
                              label="News"
                            />
                            <FormControlLabel
                              value="clickbait"
                              control={<Radio />}
                              label="Clickbait"
                            />
                          </RadioGroup>

                          <Button
                            size="small"
                            variant="contained"
                            onClick={handleSubmit}
                            color="secondary"
                          >
                            Send
                          </Button>
                        </FormControl>
                      </CardActions>
                    </Card>
                  </ThemeProvider>
                </Box>
              </Box>
            </Grid>
          </>
        )}
      </Grid>
    </Container>
  );
}

export default App;
